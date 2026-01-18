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
def TransformIf(r, expr):
    """Disables the projection key if the flag name filter expr is false.

  Args:
    r: A JSON-serializable object.
    expr: A command flag filter name expression. See `gcloud topic filters` for
      details on filter expressions. The expression variables are flag names
      without the leading *--* prefix and dashes replaced by underscores.

  Example:
    `table(name, value.if(NOT short_format))`:::
    Lists a value column if the *--short-format* command line flag is not
    specified.

  Returns:
    r
  """
    _ = expr
    return r