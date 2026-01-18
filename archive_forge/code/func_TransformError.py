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
def TransformError(r, message=None):
    """Raises an Error exception that does not generate a stack trace.

  Args:
    r: A JSON-serializable object.
    message: An error message. If not specified then the resource is formatted
      as the error message.

  Raises:
    Error: This will not generate a stack trace.
  """
    if message is None:
        message = six.text_type(r)
    raise resource_exceptions.Error(message)