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
def TransformJoin(r, sep='/', undefined=''):
    """Joins the elements of the resource list by the value of sep.

  A string resource is treated as a list of characters.

  Args:
    r: A string or list.
    sep: The separator value to use when joining.
    undefined: Returns this value if the result after joining is empty.

  Returns:
    A new string containing the resource values joined by sep.

  Example:
    `"a/b/c/d".split("/").join("!")` returns `"a!b!c!d"`.
  """
    try:
        parts = [six.text_type(i) for i in r]
        return sep.join(parts) or undefined
    except (AttributeError, TypeError):
        return undefined