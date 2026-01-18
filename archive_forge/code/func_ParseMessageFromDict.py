from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def ParseMessageFromDict(data, mapping, message, additional_fields=None):
    """Recursively generates the request message and any sub-messages.

  Args:
      data: {string: string}, A YAML like object containing the message data.
      mapping: {string: ApitoolsToKrmFieldDescriptor}, A mapping from message
        field names to mapping descriptors.
      message: The apitools class for the message.
      additional_fields: {string: object}, Additional fields to set in the
        message that are not mapped from data. Including calculated
        fields and static values.

  Returns:
    The instantiated apitools Message with all fields populated from data.

  Raises:
    InvalidDataError: If mapped fields do not exists in data.
  """
    output_message = _MapDictToApiToolsMessage(data, mapping, message)
    if additional_fields:
        for field_path, value in six.iteritems(additional_fields):
            arg_utils.SetFieldInMessage(output_message, field_path, value)
    return output_message