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
def EncryptWithPublicKey(self, public_key, secret, openssl_executable):
    """Encrypt secret with resource public key.

    Args:
      public_key: The public key for the session/cluster.
      secret: Token to be encrypted.
      openssl_executable: The path to the openssl executable.

    Returns:
      Encrypted token.
    """
    if openssl_executable:
        return self._EncodeTokenUsingOpenssl(public_key, secret, openssl_executable)
    try:
        import tink
        from tink import hybrid
    except ImportError:
        raise exceptions.PersonalAuthError('Cannot load the Tink cryptography library. Either the library is not installed, or site packages are not enabled for the Google Cloud SDK. Please consult Cloud Dataproc Personal Auth documentation on adding Tink to Google Cloud SDK for further instructions.\nhttps://cloud.google.com/dataproc/docs/concepts/iam/personal-auth')
    hybrid.register()
    context = b''
    public_key_value = json.loads(public_key)['key'][0]['keyData']['value']
    key_hash = hashlib.sha256((public_key_value + '\n').encode('utf-8')).hexdigest()
    reader = tink.JsonKeysetReader(public_key)
    kh_pub = tink.read_no_secret_keyset_handle(reader)
    encrypter = kh_pub.primitive(hybrid.HybridEncrypt)
    ciphertext = encrypter.encrypt(secret.encode('utf-8'), context)
    encoded_token = base64.b64encode(ciphertext).decode('utf-8')
    return '{}:{}'.format(key_hash, encoded_token)