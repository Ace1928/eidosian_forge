from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def ParseYamlOrJsonPolicyFile(policy_file_path, policy_message_type):
    """Create an IAM Policy protorpc.Message from a YAML or JSON formatted file.

  Returns the parsed policy object and FieldMask derived from input dict.
  Args:
    policy_file_path: Path to the YAML or JSON IAM policy file.
    policy_message_type: Policy message type to convert YAML to.

  Returns:
    a tuple of (policy, updateMask) where policy is a protorpc.Message of type
    policy_message_type filled in from the JSON or YAML policy file and
    updateMask is a FieldMask containing policy fields to be modified, based on
    which fields are present in the input file.
  Raises:
    BadFileException if the YAML or JSON file is malformed.
    IamEtagReadError if the etag is badly formatted.
  """
    policy_to_parse = yaml.load_path(policy_file_path)
    try:
        policy = encoding.PyValueToMessage(policy_message_type, policy_to_parse)
        update_mask = ','.join(sorted(policy_to_parse.keys()))
    except AttributeError as e:
        raise gcloud_exceptions.BadFileException('Policy file [{0}] is not a properly formatted YAML or JSON policy file. {1}'.format(policy_file_path, six.text_type(e)))
    except (apitools_messages.DecodeError, binascii.Error) as e:
        raise IamEtagReadError('The etag of policy file [{0}] is not properly formatted. {1}'.format(policy_file_path, six.text_type(e)))
    return (policy, update_mask)