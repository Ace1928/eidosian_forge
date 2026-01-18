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
def TransformCollection(r, undefined=''):
    """Returns the current resource collection.

  Args:
    r: A JSON-serializable object.
    undefined: This value is returned if r or the collection is empty.

  Returns:
    The current resource collection, undefined if unknown.
  """
    return undefined