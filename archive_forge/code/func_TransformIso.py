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
def TransformIso(r, undefined='T'):
    """Formats the resource to numeric ISO time format.

  Args:
    r: A JSON-serializable object.
    undefined: Returns this value if the resource does not have an isoformat()
      attribute.

  Returns:
    The numeric ISO time format for r or undefined if r is not a time.
  """
    return TransformDate(r, format='%Y-%m-%dT%H:%M:%S.%3f%Oz', undefined=undefined)