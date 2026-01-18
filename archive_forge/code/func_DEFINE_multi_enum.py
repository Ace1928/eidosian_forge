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
def DEFINE_multi_enum(name, default, enum_values, help, flag_values=_flagvalues.FLAGS, case_sensitive=True, required=False, **args):
    """Registers a flag whose value can be a list strings from enum_values.

  Use the flag on the command line multiple times to place multiple
  enum values into the list.  The 'default' may be a single string
  (which will be converted into a single-element list) or a list of
  strings.

  Args:
    name: str, the flag name.
    default: Union[Iterable[Text], Text, None], the default value of the flag;
      see `DEFINE_multi`.
    enum_values: [str], a non-empty list of strings with the possible values for
      the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    case_sensitive: Whether or not the enum is to be case-sensitive.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: Dictionary with extra keyword args that are passed to the Flag
      __init__.

  Returns:
    a handle to defined flag.
  """
    parser = _argument_parser.EnumParser(enum_values, case_sensitive)
    serializer = _argument_parser.ArgumentSerializer()
    return DEFINE_multi(parser, serializer, name, default, help, flag_values, required=required, **args)