from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.kms import get_digest
from googlecloudsdk.command_lib.kms import maps
import six
def GetKeyUri(key_ref):
    """Returns the URI used as the default for KMS keys.

  This should look something like '//cloudkms.googleapis.com/v1/...'

  Args:
    key_ref: A CryptoKeyVersion Resource.

  Returns:
    The string URI.
  """
    return key_ref.SelfLink().split(':', 1)[1]