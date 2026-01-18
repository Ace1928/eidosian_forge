from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import openssl_encryption_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as core_encoding
from googlecloudsdk.core.util import files
def _ConstructWindowsKeyEntry(self, user, modulus, exponent, email):
    """Return a JSON formatted entry for 'windows-keys'."""
    expire_str = time_util.CalculateExpiration(RSA_KEY_EXPIRATION_TIME_SEC)
    windows_key_data = {'userName': user, 'modulus': core_encoding.Decode(modulus), 'exponent': core_encoding.Decode(exponent), 'email': email, 'expireOn': expire_str}
    windows_key_entry = json.dumps(windows_key_data, sort_keys=True)
    return windows_key_entry