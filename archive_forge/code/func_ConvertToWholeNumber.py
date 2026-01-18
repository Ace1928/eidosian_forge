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
def ConvertToWholeNumber(amount, unit):
    """Convert input value and units to a whole number of a lower unit.

  Args:
    amount: str, a number, for example '3.25'
    unit: str, a binary prefix, for example 'GB' or 'GiB'

  Returns:
    (decimal.Decimal(), str), a tuple of number and suffix, converted such that
    the number returned is an integer, or the value, in Bytes, of the amount
    input. For example (23, 'MiB'). Note that IEC binary prefixes are always
    assumed and returned.
  """
    return_amount = decimal.Decimal(amount)
    return_unit = unit
    while not float(return_amount).is_integer() and return_unit and (return_unit in _UnitToLowerUnitDict):
        return_amount, return_unit = (return_amount * 1024, _UnitToLowerUnitDict[return_unit])
    return (return_amount, return_unit)