import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
class SaslGSSAPIAuthenticator(BaseSaslAuthenticator):

    def __init__(self, *, loop, principal):
        self._loop = loop
        self._principal = principal
        self._authenticator = self.authenticator_gssapi()

    def authenticator_gssapi(self):
        name = gssapi.Name(self._principal, name_type=gssapi.NameType.hostbased_service)
        cname = name.canonicalize(gssapi.MechType.kerberos)
        client_ctx = gssapi.SecurityContext(name=cname, usage='initiate')
        server_token = None
        while not client_ctx.complete:
            client_token = client_ctx.step(server_token)
            client_token = client_token or b''
            server_token = (yield (client_token, True))
        msg = client_ctx.unwrap(server_token).message
        qop = struct.pack('b', SASL_QOP_AUTH & msg[0])
        msg = qop + msg[1:]
        msg = client_ctx.wrap(msg + self._principal.encode(), False).message
        yield (msg, False)