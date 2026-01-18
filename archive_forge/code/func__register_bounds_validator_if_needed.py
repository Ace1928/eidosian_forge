from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import types
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
def _register_bounds_validator_if_needed(parser, name, flag_values):
    """Enforces lower and upper bounds for numeric flags.

  Args:
    parser: NumericParser (either FloatParser or IntegerParser), provides lower
      and upper bounds, and help text to display.
    name: str, name of the flag
    flag_values: FlagValues.
  """
    if parser.lower_bound is not None or parser.upper_bound is not None:

        def checker(value):
            if value is not None and parser.is_outside_bounds(value):
                message = '%s is not %s' % (value, parser.syntactic_help)
                raise _exceptions.ValidationError(message)
            return True
        _validators.register_validator(name, checker, flag_values=flag_values)