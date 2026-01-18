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
def DEFINE_list(name, default, help, flag_values=_flagvalues.FLAGS, required=False, **args):
    """Registers a flag whose value is a comma-separated list of strings.

  The flag value is parsed with a CSV parser.

  Args:
    name: str, the flag name.
    default: list|str|None, the default value of the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: Dictionary with extra keyword args that are passed to the Flag
      __init__.

  Returns:
    a handle to defined flag.
  """
    parser = _argument_parser.ListParser()
    serializer = _argument_parser.CsvListSerializer(',')
    return DEFINE(parser, name, default, help, flag_values, serializer, required=required, **args)