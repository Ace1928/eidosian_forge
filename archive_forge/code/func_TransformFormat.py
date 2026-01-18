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
def TransformFormat(r, projection, fmt, *args):
    """Formats resource key values.

  Args:
    r: A JSON-serializable object.
    projection: The parent ProjectionSpec.
    fmt: The format string with {0} ... {nargs-1} references to the resource
      attribute name arg values.
    *args: The resource attribute key expression to format. The printer
      projection symbols and aliases may be used in key expressions. If no args
      are specified then the resource is used as the arg list if it is a list,
      otherwise the resource is used as the only arg.

  Returns:
    The formatted string.

  Example:
    `--format='value(format("{0:f.1}/{1:f.1}", q.CPU.default, q.CPU.limit))'`:::
    Formats q.CPU.default and q.CPU.limit as floating point numbers.
  """
    if args:
        columns = projection.compiler('({0})'.format(','.join(args)), by_columns=True, defaults=projection).Evaluate(r)
    elif isinstance(r, list):
        columns = r
    else:
        columns = [r or '']
    return fmt.format(*columns)