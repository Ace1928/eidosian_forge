from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from six.moves import range  # pylint: disable=redefined-builtin
def IsListLike(resource):
    """Checks if resource is a list-like iterable object.

  Args:
    resource: The object to check.

  Returns:
    True if resource is a list-like iterable object.
  """
    return isinstance(resource, list) or (hasattr(resource, '__iter__') and (hasattr(resource, 'next') or hasattr(resource, '__next__')))