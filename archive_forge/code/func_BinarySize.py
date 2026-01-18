import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def BinarySize(lower_bound=None, upper_bound=None, suggested_binary_size_scales=None, default_unit='G', type_abbr='B'):
    """Returns a function that can parse binary sizes.

  Binary sizes are defined as base-2 values representing number of
  bytes.

  Input to the parsing function must be a string of the form:

    DECIMAL[UNIT]

  The amount must be non-negative. Valid units are "B", "KB", "MB",
  "GB", "TB", "PB", "KiB", "MiB", "GiB", "TiB", "PiB".  If the unit is
  omitted then default_unit is assumed.

  The result is parsed in bytes. For example:

    parser = BinarySize()
    assert parser('10GB') == 1073741824

  Another example:

    parser = BinarySize()
    assert parser('2.5KB') == 2560

  Args:
    lower_bound: str, An inclusive lower bound for values.
    upper_bound: str, An inclusive upper bound for values.
    suggested_binary_size_scales: list, A list of strings with units that will
      be recommended to user.
    default_unit: str, unit used when user did not specify unit.
    type_abbr: str, the type suffix abbreviation, e.g., B for bytes, b/s for
      bits/sec.

  Raises:
    ArgumentTypeError: If either the lower_bound or upper_bound
      cannot be parsed. The returned function will also raise this
      error if it cannot parse its input. This exception is also
      raised if the returned function receives an out-of-bounds
      input.

  Returns:
    A function that accepts a single binary size as input to be
      parsed.
  """
    return _ValueParser(_BINARY_SIZE_SCALES, default_unit=default_unit, lower_bound=lower_bound, upper_bound=upper_bound, strict_case=False, type_abbr=type_abbr, suggested_binary_size_scales=suggested_binary_size_scales)