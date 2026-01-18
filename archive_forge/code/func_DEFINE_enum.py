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
def DEFINE_enum(name, default, enum_values, help, flag_values=_flagvalues.FLAGS, module_name=None, required=False, **args):
    """Registers a flag whose value can be any string from enum_values.

  Instead of a string enum, prefer `DEFINE_enum_class`, which allows
  defining enums from an `enum.Enum` class.

  Args:
    name: str, the flag name.
    default: str|None, the default value of the flag.
    enum_values: [str], a non-empty list of strings with the possible values for
      the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag. If not
      provided, it will be computed using the stack trace of this call.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: dict, the extra keyword args that are passed to Flag __init__.

  Returns:
    a handle to defined flag.
  """
    return DEFINE_flag(_flag.EnumFlag(name, default, help, enum_values, **args), flag_values, module_name, required)