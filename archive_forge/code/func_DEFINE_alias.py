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
def DEFINE_alias(name, original_name, flag_values=_flagvalues.FLAGS, module_name=None):
    """Defines an alias flag for an existing one.

  Args:
    name: str, the flag name.
    original_name: str, the original flag name.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: A string, the name of the module that defines this flag.

  Returns:
    a handle to defined flag.

  Raises:
    flags.FlagError:
      UnrecognizedFlagError: if the referenced flag doesn't exist.
      DuplicateFlagError: if the alias name has been used by some existing flag.
  """
    if original_name not in flag_values:
        raise _exceptions.UnrecognizedFlagError(original_name)
    flag = flag_values[original_name]

    class _FlagAlias(_flag.Flag):
        """Overrides Flag class so alias value is copy of original flag value."""

        def parse(self, argument):
            flag.parse(argument)
            self.present += 1

        def _parse_from_default(self, value):
            return value

        @property
        def value(self):
            return flag.value

        @value.setter
        def value(self, value):
            flag.value = value
    help_msg = 'Alias for --%s.' % flag.name
    return DEFINE_flag(_FlagAlias(flag.parser, flag.serializer, name, flag.default, help_msg, boolean=flag.boolean), flag_values, module_name)