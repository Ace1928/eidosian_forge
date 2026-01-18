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
def FieldHelpDocs(message, section='Fields'):
    """Gets the help text for the fields in the request message.

  Args:
    message: The apitools message.
    section: str, The section to extract help data from. Fields is the default,
      may also be Values to extract enum data, for example.

  Returns:
    {str: str}, A mapping of field name to help text.
  """
    field_helps = {}
    current_field = None
    match = re.search('^\\s+{}:.*$'.format(section), message.__doc__ or '', re.MULTILINE)
    if not match:
        return field_helps
    for line in message.__doc__[match.end():].splitlines():
        match = re.match('^\\s+(\\w+): (.*)$', line)
        if match:
            current_field = match.group(1)
            field_helps[current_field] = match.group(2).strip()
        elif current_field:
            to_append = line.strip()
            if to_append:
                current_text = field_helps.get(current_field, '')
                field_helps[current_field] = current_text + ' ' + to_append
    return field_helps