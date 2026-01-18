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
def DEFINE_multi(parser, serializer, name, default, help, flag_values=_flagvalues.FLAGS, module_name=None, required=False, **args):
    """Registers a generic MultiFlag that parses its args with a given parser.

  Auxiliary function.  Normal users should NOT use it directly.

  Developers who need to create their own 'Parser' classes for options
  which can appear multiple times can call this module function to
  register their flags.

  Args:
    parser: ArgumentParser, used to parse the flag arguments.
    serializer: ArgumentSerializer, the flag serializer instance.
    name: str, the flag name.
    default: Union[Iterable[T], Text, None], the default value of the flag. If
      the value is text, it will be parsed as if it was provided from the
      command line. If the value is a non-string iterable, it will be iterated
      over to create a shallow copy of the values. If it is None, it is left
      as-is.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: A string, the name of the Python module declaring this flag. If
      not provided, it will be computed using the stack trace of this call.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: Dictionary with extra keyword args that are passed to the Flag
      __init__.

  Returns:
    a handle to defined flag.
  """
    return DEFINE_flag(_flag.MultiFlag(parser, serializer, name, default, help, **args), flag_values, module_name, required)