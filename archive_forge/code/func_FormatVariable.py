from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import socket
from apitools.base.py import encoding
from googlecloudsdk.api_lib.runtime_config import exceptions as rtc_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as sdk_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def FormatVariable(message, output_value=False):
    """Returns the variable message as a dict with a shortened name.

  This method first converts the variable message to a dict with a shortened
  name and an atomicName. Then, decodes the variable value in the dict if the
  output_value flag is True.

  Args:
    message: A protorpclite message.
    output_value: A bool flag indicates whether we want to decode and output the
        values of the variables. The default value of this flag is False.

  Returns:
    A dict representation of the message with a shortened name field.
  """
    message_dict = _DictWithShortName(message, lambda name: '/'.join(name.split('/')[VARIABLE_NAME_PREFIX_LENGTH:]))
    if output_value:
        if 'text' in message_dict:
            message_dict['value'] = message_dict['text']
        else:
            message_dict['value'] = base64.b64decode(message_dict['value'])
    return message_dict