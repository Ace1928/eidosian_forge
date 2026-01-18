from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
class FormattableErrorPayload(string.Formatter):
    """Generic payload for an HTTP error that supports format strings.

  Attributes:
    content: The dumped JSON content.
    message: The human readable error message.
    status_code: The HTTP status code number.
    status_description: The status_code description.
    status_message: Context specific status message.
  """

    def __init__(self, http_error):
        """Initialize a FormattableErrorPayload instance.

    Args:
      http_error: An Exception that subclasses can use to populate class
        attributes, or a string to use as the error message.
    """
        super(FormattableErrorPayload, self).__init__()
        self._value = '{?}'
        self.content = {}
        self.status_code = 0
        self.status_description = ''
        self.status_message = ''
        if isinstance(http_error, six.string_types):
            self.message = http_error
        else:
            self.message = self._MakeGenericMessage()

    def get_field(self, field_name, unused_args, unused_kwargs):
        """Returns the value of field_name for string.Formatter.format().

    Args:
      field_name: The format string field name to get in the form
        name - the value of name in the payload, '' if undefined
        name?FORMAT - if name is non-empty then re-formats with FORMAT, where
          {?} is the value of name. For example, if name=NAME then
          {name?\\nname is "{?}".} expands to '\\nname is "NAME".'.
        .a.b.c - the value of a.b.c in the JSON decoded payload contents.
          For example, '{.errors.reason?[{?}]}' expands to [REASON] if
          .errors.reason is defined.
      unused_args: Ignored.
      unused_kwargs: Ignored.

    Returns:
      The value of field_name for string.Formatter.format().
    """
        field_name = _Expand(field_name)
        if field_name == '?':
            return (self._value, field_name)
        parts = field_name.split('?', 1)
        subparts = parts.pop(0).split(':', 1)
        name = subparts.pop(0)
        printer_format = subparts.pop(0) if subparts else None
        recursive_format = parts.pop(0) if parts else None
        name, value = self._GetField(name)
        if not value and (not isinstance(value, (int, float))):
            return ('', name)
        if printer_format or not isinstance(value, (six.text_type, six.binary_type, float) + six.integer_types):
            buf = io.StringIO()
            resource_printer.Print(value, printer_format or 'default', out=buf, single=True)
            value = buf.getvalue().strip()
        if recursive_format:
            self._value = value
            value = self.format(_Expand(recursive_format))
        return (value, name)

    def _GetField(self, name):
        """Gets the value corresponding to name in self.content or class attributes.

    If `name` starts with a period, treat it as a key in self.content and get
    the corresponding value. Otherwise get the value of the class attribute
    named `name` first and fall back to checking keys in self.content.

    Args:
      name (str): The name of the attribute to return the value of.

    Returns:
      A tuple where the first value is `name` with any leading periods dropped,
      and the second value is the value of a class attribute or key in
      self.content.
    """
        if '.' in name:
            if name.startswith('.'):
                check_payload_attributes = False
                name = name[1:]
            else:
                check_payload_attributes = True
            key = resource_lex.Lexer(name).Key()
            content = self.content
            if check_payload_attributes and key:
                value = self.__dict__.get(key[0], None)
                if value:
                    content = {key[0]: value}
            value = resource_property.Get(content, key, None)
        elif name:
            value = self.__dict__.get(name, None)
        else:
            value = None
        return (name, value)

    def _MakeGenericMessage(self):
        """Makes a generic human readable message from the HttpError."""
        description = self._MakeDescription()
        if self.status_message:
            return '{0}: {1}'.format(description, self.status_message)
        return description

    def _MakeDescription(self):
        """Makes description for error by checking which fields are filled in."""
        description = self.status_description
        if description:
            if description.endswith('.'):
                description = description[:-1]
            return description
        return 'HTTPError {0}'.format(self.status_code)