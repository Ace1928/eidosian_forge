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
def TransformLen(r):
    """Returns the length of the resource if it is non-empty, 0 otherwise.

  Args:
    r: A JSON-serializable object.

  Returns:
    The length of r if r is non-empty, 0 otherwise.
  """
    try:
        return len(r)
    except TypeError:
        return 0