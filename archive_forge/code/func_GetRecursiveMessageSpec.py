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
def GetRecursiveMessageSpec(message, definitions=None):
    """Gets the recursive representation of a message as a dictionary.

  Args:
    message: The apitools message.
    definitions: A list of message definitions already encountered.

  Returns:
    {str: object}, A recursive mapping of field name to its data.
  """
    if definitions is None:
        definitions = []
    if message in definitions:
        return {}
    definitions.append(message)
    field_helps = FieldHelpDocs(message)
    data = {}
    for field in message.all_fields():
        field_data = {'description': field_helps.get(field.name)}
        field_data['repeated'] = field.repeated
        if field.variant == messages.Variant.MESSAGE:
            field_data['type'] = field.type.__name__
            fields = GetRecursiveMessageSpec(field.type, definitions=definitions)
            if fields:
                field_data['fields'] = fields
        else:
            field_data['type'] = field.variant
            if field.variant == messages.Variant.ENUM:
                enum_help = FieldHelpDocs(field.type, 'Values')
                field_data['choices'] = {n: enum_help.get(n) for n in field.type.names()}
        data[field.name] = field_data
    definitions.pop()
    return data