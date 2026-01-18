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
def _DictWithShortName(message, name_converter):
    """Returns a dict representation of the message with a shortened name value.

  This method does three things:
  1. converts message to a dict.
  2. shortens the value of the name field using name_converter
  3. sets atomicName to the original value of name.

  Args:
    message: A protorpclite message.
    name_converter: A function that takes an atomic name as a parameter and
        returns a shortened name.

  Returns:
    A dict representation of the message with a shortened name field.

  Raises:
    ValueError: If the original message already contains an atomicName field.
  """
    message_dict = encoding.MessageToDict(message)
    if 'name' in message_dict:
        if 'atomicName' in message_dict:
            raise ValueError('Original message cannot contain an atomicName field.')
        message_dict['atomicName'] = message_dict['name']
        message_dict['name'] = name_converter(message_dict['name'])
    return message_dict