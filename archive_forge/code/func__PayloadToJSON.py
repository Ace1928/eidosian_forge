from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.core import log as logging
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _PayloadToJSON(payload, is_json_payload=False):
    """Converts the property into a JSON string.

  This is mainly used for properties that have additional properties under them.
  For example, the resource and split properties of LogEntry store objects
  containing additional fields. These fields must be elementary and have no
  nested objects within.

  Args:
    payload: the property to serialize to JSON.
    is_json_payload: only used for the jsonPayload property since the values
                     values are nested in an additional string_value attribute.

  Returns:
    The JSON string representation of the provided property of a LogEntry.
  """
    payload_string = '{\n'
    properties = payload.additionalProperties
    length = len(properties)
    for i in range(length):
        field = properties[i]
        payload_string += '"{}": "{}"'.format(field.key, field.value.string_value if is_json_payload else field.value) + ('\n' if i == length - 1 else ',\n')
    payload_string += '}'
    return payload_string