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
def TransformCount(r):
    """Counts the number of each item in the list.

  A string resource is treated as a list of characters.

  Args:
    r: A string or list.

  Returns:
    A dictionary mapping list elements to the number of them in the input list.

  Example:
    `"b/a/b/c".split("/").count()` returns `{a: 1, b: 2, c: 1}`.
  """
    if not r:
        return {}
    try:
        count = {}
        for item in r:
            c = count.get(item, 0)
            count[item] = c + 1
        return count
    except TypeError:
        return {}