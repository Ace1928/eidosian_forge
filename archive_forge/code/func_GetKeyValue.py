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
def GetKeyValue(r, key, undefined=None):
    """Returns the value for key in r.

  Args:
    r: The resource object.
    key: The dotted attribute name string.
    undefined: This is returned if key is not in r.

  Returns:
    The value for key in r.
  """
    return resource_property.Get(r, _GetParsedKey(key), undefined)