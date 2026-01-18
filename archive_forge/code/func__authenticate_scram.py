from __future__ import annotations
import functools
import hashlib
import hmac
import os
import socket
import typing
from base64 import standard_b64decode, standard_b64encode
from collections import namedtuple
from typing import (
from urllib.parse import quote
from bson.binary import Binary
from bson.son import SON
from pymongo.auth_aws import _authenticate_aws
from pymongo.auth_oidc import _authenticate_oidc, _get_authenticator, _OIDCProperties
from pymongo.errors import ConfigurationError, OperationFailure
from pymongo.saslprep import saslprep
def _authenticate_scram(credentials: MongoCredential, conn: Connection, mechanism: str) -> None:
    """Authenticate using SCRAM."""
    username = credentials.username
    if mechanism == 'SCRAM-SHA-256':
        digest = 'sha256'
        digestmod = hashlib.sha256
        data = saslprep(credentials.password).encode('utf-8')
    else:
        digest = 'sha1'
        digestmod = hashlib.sha1
        data = _password_digest(username, credentials.password).encode('utf-8')
    source = credentials.source
    cache = credentials.cache
    _hmac = hmac.HMAC
    ctx = conn.auth_ctx
    if ctx and ctx.speculate_succeeded():
        assert isinstance(ctx, _ScramContext)
        assert ctx.scram_data is not None
        nonce, first_bare = ctx.scram_data
        res = ctx.speculative_authenticate
    else:
        nonce, first_bare, cmd = _authenticate_scram_start(credentials, mechanism)
        res = conn.command(source, cmd)
    assert res is not None
    server_first = res['payload']
    parsed = _parse_scram_response(server_first)
    iterations = int(parsed[b'i'])
    if iterations < 4096:
        raise OperationFailure('Server returned an invalid iteration count.')
    salt = parsed[b's']
    rnonce = parsed[b'r']
    if not rnonce.startswith(nonce):
        raise OperationFailure('Server returned an invalid nonce.')
    without_proof = b'c=biws,r=' + rnonce
    if cache.data:
        client_key, server_key, csalt, citerations = cache.data
    else:
        client_key, server_key, csalt, citerations = (None, None, None, None)
    if not client_key or salt != csalt or iterations != citerations:
        salted_pass = hashlib.pbkdf2_hmac(digest, data, standard_b64decode(salt), iterations)
        client_key = _hmac(salted_pass, b'Client Key', digestmod).digest()
        server_key = _hmac(salted_pass, b'Server Key', digestmod).digest()
        cache.data = (client_key, server_key, salt, iterations)
    stored_key = digestmod(client_key).digest()
    auth_msg = b','.join((first_bare, server_first, without_proof))
    client_sig = _hmac(stored_key, auth_msg, digestmod).digest()
    client_proof = b'p=' + standard_b64encode(_xor(client_key, client_sig))
    client_final = b','.join((without_proof, client_proof))
    server_sig = standard_b64encode(_hmac(server_key, auth_msg, digestmod).digest())
    cmd = SON([('saslContinue', 1), ('conversationId', res['conversationId']), ('payload', Binary(client_final))])
    res = conn.command(source, cmd)
    parsed = _parse_scram_response(res['payload'])
    if not hmac.compare_digest(parsed[b'v'], server_sig):
        raise OperationFailure('Server returned an invalid signature.')
    if not res['done']:
        cmd = SON([('saslContinue', 1), ('conversationId', res['conversationId']), ('payload', Binary(b''))])
        res = conn.command(source, cmd)
        if not res['done']:
            raise OperationFailure('SASL conversation failed to complete.')