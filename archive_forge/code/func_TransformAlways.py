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
def TransformAlways(r):
    """Marks a transform sequence to always be applied.

  In some cases transforms are disabled. Prepending always() to a transform
  sequence causes the sequence to always be evaluated.

  Example:
    `some_field.always().foo().bar()`:::
    Always applies foo() and then bar().

  Args:
    r: A resource.

  Returns:
    r.
  """
    return r