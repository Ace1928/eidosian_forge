from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
@retry.RetryOnException(max_retrials=1)
def _EncodeTokenUsingOpenssl(self, public_key, secret, openssl_executable):
    """Encode token using OpenSSL.

    Args:
      public_key: The public key for the session/cluster.
      secret: Token to be encrypted.
      openssl_executable: The path to the openssl executable.

    Returns:
      Encrypted token.
    """
    key_hash = hashlib.sha256((public_key + '\n').encode('utf-8')).hexdigest()
    iv_bytes = base64.b16encode(os.urandom(16))
    initialization_vector = iv_bytes.decode('utf-8')
    initial_key = os.urandom(32)
    encryption_key = self._DeriveHkdfKey(initial_key, 'encryption_key'.encode('utf-8'), openssl_executable)
    auth_key = base64.b16encode(self._DeriveHkdfKey(initial_key, 'auth_key'.encode('utf-8'), openssl_executable)).decode('utf-8')
    with tempfile.NamedTemporaryFile() as kf:
        kf.write(public_key.encode('utf-8'))
        kf.seek(0)
        encrypted_key = self._RunOpensslCommand(openssl_executable, ['rsautl', '-oaep', '-encrypt', '-pubin', '-inkey', kf.name], stdin=base64.b64encode(initial_key))
    if len(encrypted_key) != 512:
        raise ValueError('The encrypted key is expected to be 512 bytes long.')
    encoded_key = base64.b64encode(encrypted_key).decode('utf-8')
    with tempfile.NamedTemporaryFile() as pf:
        pf.write(encryption_key)
        pf.seek(0)
        encrypt_args = ['enc', '-aes-256-ctr', '-salt', '-iv', initialization_vector, '-pass', 'file:{}'.format(pf.name)]
        encrypted_token = self._RunOpensslCommand(openssl_executable, encrypt_args, stdin=secret.encode('utf-8'))
    if len(encrypted_key) != 512:
        raise ValueError('The encrypted key is expected to be 512 bytes long.')
    encoded_token = base64.b64encode(encrypted_token).decode('utf-8')
    hmac_input = bytearray(iv_bytes)
    hmac_input.extend(encrypted_token)
    hmac_tag = self._ComputeHmac(auth_key, hmac_input, openssl_executable).decode('utf-8')[0:32]
    return '{}:{}:{}:{}:{}'.format(key_hash, encoded_token, encoded_key, initialization_vector, hmac_tag)