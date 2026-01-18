from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def GeneratePublicKeyDataFromFile(path):
    """Generate public key data from a path.

  Args:
    path: (bytes) the public key file path given by the command.

  Raises:
    InvalidArgumentException: if the public key file path provided does not
                              exist or is too large.
  Returns:
    A public key encoded using the UTF-8 charset.
  """
    try:
        public_key_data = arg_parsers.FileContents()(path).strip()
    except arg_parsers.ArgumentTypeError as e:
        raise gcloud_exceptions.InvalidArgumentException('public_key_file', '{}. Please double check your input and try again.'.format(e))
    return public_key_data.encode('utf-8')