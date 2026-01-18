from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def GetFieldValueFromMessage(message, field_path):
    """Extract the value of the field given a dotted field path.

  If the field_path does not exist, an error is logged.

  Args:
    message: The apitools message to dig into.
    field_path: str, The dotted path of attributes and sub-attributes.

  Raises:
    InvalidFieldPathError: When the path is invalid.

  Returns:
    The value or if not set, None.
  """
    root_message = message
    fields = field_path.split('.')
    for i, f in enumerate(fields):
        index_found = re.match('(.+)\\[(\\d+)\\]$', f)
        if index_found:
            f, index = index_found.groups()
            index = int(index)
        else:
            index = None
        try:
            field = message.field_by_name(f)
        except KeyError:
            raise InvalidFieldPathError(field_path, root_message, UnknownFieldError(f, message))
        if index_found:
            if not field.repeated:
                raise InvalidFieldPathError(field_path, root_message, 'Index cannot be specified for non-repeated field [{}]'.format(f))
        elif field.repeated and i < len(fields) - 1:
            raise InvalidFieldPathError(field_path, root_message, 'Index needs to be specified for repeated field [{}]'.format(f))
        message = getattr(message, f)
        if message and index_found:
            message = message[index] if index < len(message) else None
        if not message and i < len(fields) - 1:
            if isinstance(field, messages.MessageField):
                message = field.type()
            else:
                raise InvalidFieldPathError(field_path, root_message, '[{}] is not a valid field on field [{}]'.format(f, field.type.__name__))
    return message