from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
def GetKeyPair(self, key_length=DEFAULT_KEY_LENGTH):
    """Returns an RSA key pair (private key)."""
    return self.RunOpenSSL(['genrsa', six.text_type(key_length)])