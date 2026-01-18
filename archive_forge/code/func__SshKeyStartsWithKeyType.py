from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _SshKeyStartsWithKeyType(key):
    """Checks if the key starts with any key type in constants.SSH_KEY_TYPES.

  Args:
    key: A ssh key in metadata.

  Returns:
    True if the key starts with any key type in constants.SSH_KEY_TYPES, returns
    false otherwise.

  """
    key_starts_with_types = [key.startswith(key_type) for key_type in constants.SSH_KEY_TYPES]
    return any(key_starts_with_types)