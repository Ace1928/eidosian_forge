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
def _ValueParser(scales, default_unit, lower_bound=None, upper_bound=None, strict_case=True, type_abbr='B', suggested_binary_size_scales=None):
    """A helper that returns a function that can parse values with units.

  Casing for all units matters.

  Args:
    scales: {str: int}, A dictionary mapping units to their magnitudes in
      relation to the lowest magnitude unit in the dict.
    default_unit: str, The default unit to use if the user's input is missing
      unit.
    lower_bound: str, An inclusive lower bound.
    upper_bound: str, An inclusive upper bound.
    strict_case: bool, whether to be strict on case-checking
    type_abbr: str, the type suffix abbreviation, e.g., B for bytes, b/s for
      bits/sec.
    suggested_binary_size_scales: list, A list of strings with units that will
      be recommended to user.

  Returns:
    A function that can parse values.
  """

    def UnitsByMagnitude(suggested_binary_size_scales=None):
        """Returns a list of the units in scales sorted by magnitude."""
        scale_items = sorted(six.iteritems(scales), key=lambda value: (value[1], value[0]))
        if suggested_binary_size_scales is None:
            return [key + type_abbr for key, _ in scale_items]
        return [key + type_abbr for key, _ in scale_items if key + type_abbr in suggested_binary_size_scales]

    def Parse(value):
        """Parses value that can contain a unit and type abbreviation."""
        match = re.match(_VALUE_PATTERN, value, re.VERBOSE)
        if not match:
            raise ArgumentTypeError(_GenerateErrorMessage(InvalidInputErrorMessage(UnitsByMagnitude(suggested_binary_size_scales)), user_input=value))
        suffix = match.group('suffix') or ''
        amount, suffix = ConvertToWholeNumber(match.group('amount'), suffix)
        if not float(amount).is_integer():
            raise ArgumentTypeError(_GenerateErrorMessage(InvalidInputErrorMessage(UnitsByMagnitude(suggested_binary_size_scales)), user_input=value))
        amount = int(amount)
        unit = _DeleteTypeAbbr(suffix, type_abbr)
        if strict_case:
            unit_case = unit
            default_unit_case = _DeleteTypeAbbr(default_unit, type_abbr)
            scales_case = scales
        else:
            unit_case = unit.upper()
            default_unit_case = _DeleteTypeAbbr(default_unit.upper(), type_abbr)
            scales_case = dict([(k.upper(), v) for k, v in scales.items()])
        if not unit and unit == suffix:
            return amount * scales_case[default_unit_case]
        elif unit_case in scales_case:
            return amount * scales_case[unit_case]
        else:
            raise ArgumentTypeError(_GenerateErrorMessage('unit must be one of {0}'.format(', '.join(UnitsByMagnitude())), user_input=unit))
    if lower_bound is None:
        parsed_lower_bound = None
    else:
        parsed_lower_bound = Parse(lower_bound)
    if upper_bound is None:
        parsed_upper_bound = None
    else:
        parsed_upper_bound = Parse(upper_bound)

    def ParseWithBoundsChecking(value):
        """Same as Parse except bound checking is performed."""
        if value is None:
            return None
        else:
            parsed_value = Parse(value)
            if parsed_lower_bound is not None and parsed_value < parsed_lower_bound:
                raise ArgumentTypeError(_GenerateErrorMessage('value must be greater than or equal to {0}'.format(lower_bound), user_input=value))
            elif parsed_upper_bound is not None and parsed_value > parsed_upper_bound:
                raise ArgumentTypeError(_GenerateErrorMessage('value must be less than or equal to {0}'.format(upper_bound), user_input=value))
            else:
                return parsed_value
    return ParseWithBoundsChecking