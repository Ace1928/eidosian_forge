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
def DEFINE_multi_enum_class(name, default, enum_class, help, flag_values=_flagvalues.FLAGS, module_name=None, case_sensitive=False, required=False, **args):
    """Registers a flag whose value can be a list of enum members.

  Use the flag on the command line multiple times to place multiple
  enum values into the list.

  Args:
    name: str, the flag name.
    default: Union[Iterable[Enum], Iterable[Text], Enum, Text, None], the
      default value of the flag; see `DEFINE_multi`; only differences are
      documented here. If the value is a single Enum, it is treated as a
      single-item list of that Enum value. If it is an iterable, text values
      within the iterable will be converted to the equivalent Enum objects.
    enum_class: class, the Enum class with all the possible values for the flag.
        help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: A string, the name of the Python module declaring this flag. If
      not provided, it will be computed using the stack trace of this call.
    case_sensitive: bool, whether to map strings to members of the enum_class
      without considering case.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: Dictionary with extra keyword args that are passed to the Flag
      __init__.

  Returns:
    a handle to defined flag.
  """
    return DEFINE_flag(_flag.MultiEnumClassFlag(name, default, help, enum_class, case_sensitive=case_sensitive), flag_values, module_name, required=required, **args)