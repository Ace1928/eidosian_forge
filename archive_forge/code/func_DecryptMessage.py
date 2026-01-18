from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
def DecryptMessage(self, key, enc_message, destroy_key=False):
    """Returns a decrypted message using the provided key.

    Args:
      key: An openssl key pair (private key).
      enc_message: a base64 encoded encrypted message.
      destroy_key: Unused for OpenSSL.
    Returns:
      Decrypted version of enc_message
    """
    del destroy_key
    encrypted_message_data = base64.b64decode(enc_message)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(key)
        tf.flush()
        openssl_args = ['rsautl', '-decrypt', '-oaep', '-inkey', tf.name]
        message = self.RunOpenSSL(openssl_args, cmd_input=encrypted_message_data)
    return message