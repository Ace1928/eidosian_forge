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
def TransformUri(r, undefined='.'):
    """Gets the resource URI.

  Args:
    r: A JSON-serializable object.
    undefined: Returns this if a the URI for r cannot be determined.

  Returns:
    The URI for r or undefined if not defined.
  """

    def _GetAttr(attr):
        """Returns the string value for attr or None if the value is not a string.

    Args:
      attr: The attribute object to get the value from.

    Returns:
      The string value for attr or None if the value is not a string.
    """
        try:
            attr = attr()
        except TypeError:
            pass
        return attr if isinstance(attr, six.string_types) else None
    if isinstance(r, six.string_types):
        if r.startswith('https://'):
            return r
    elif r:
        for name in ('selfLink', 'SelfLink'):
            uri = _GetAttr(resource_property.Get(r, [name], None))
            if uri:
                return uri
    return undefined