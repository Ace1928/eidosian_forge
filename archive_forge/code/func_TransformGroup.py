from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformGroup(r, *keys):
    """Formats a [...] grouped list.

  Each group is enclosed in [...]. The first item separator is ':', subsequent
  separators are ','.
    [item1] [item1] ...
    [item1: item2] ... [item1: item2]
    [item1: item2, item3] ... [item1: item2, item3]

  Args:
    r: A JSON-serializable object.
    *keys: Optional attribute keys to select from the list. Otherwise
      the string value of each list item is selected.

  Returns:
    The [...] grouped formatted list, [] if r is empty.
  """
    if not r:
        return '[]'
    buf = io.StringIO()
    sep = None
    parsed_keys = [_GetParsedKey(key) for key in keys]
    for item in r:
        if sep:
            buf.write(sep)
        else:
            sep = ' '
        if not parsed_keys:
            buf.write('[{0}]'.format(six.text_type(item)))
        else:
            buf.write('[')
            sub = None
            for key in parsed_keys:
                if sub:
                    buf.write(sub)
                    sub = ', '
                else:
                    sub = ': '
                value = resource_property.Get(item, key, None)
                if value is not None:
                    buf.write(six.text_type(value))
            buf.write(']')
    return buf.getvalue()