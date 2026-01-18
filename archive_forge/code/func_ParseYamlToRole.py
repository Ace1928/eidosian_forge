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
def ParseYamlToRole(file_path, role_message_type):
    """Construct an IAM Role protorpc.Message from a Yaml formatted file.

  Args:
    file_path: Path to the Yaml IAM Role file.
    role_message_type: Role message type to convert Yaml to.

  Returns:
    a protorpc.Message of type role_message_type filled in from the Yaml
    role file.
  Raises:
    BadFileException if the Yaml file is malformed or does not exist.
  """
    role_to_parse = yaml.load_path(file_path)
    if 'stage' in role_to_parse:
        role_to_parse['stage'] = role_to_parse['stage'].upper()
    try:
        role = encoding.PyValueToMessage(role_message_type, role_to_parse)
    except AttributeError as e:
        raise gcloud_exceptions.BadFileException('Role file {0} is not a properly formatted YAML role file. {1}'.format(file_path, six.text_type(e)))
    except (apitools_messages.DecodeError, binascii.Error) as e:
        raise IamEtagReadError('The etag of role file {0} is not properly formatted. {1}'.format(file_path, six.text_type(e)))
    return role