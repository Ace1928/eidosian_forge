from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.flags import _exceptions
class SingleFlagValidator(Validator):
    """Validator behind register_validator() method.

  Validates that a single flag passes its checker function. The checker function
  takes the flag value and returns True (if value looks fine) or, if flag value
  is not valid, either returns False or raises an Exception.
  """

    def __init__(self, flag_name, checker, message):
        """Constructor.

    Args:
      flag_name: string, name of the flag.
      checker: function to verify the validator.
          input  - value of the corresponding flag (string, boolean, etc).
          output - bool, True if validator constraint is satisfied.
              If constraint is not satisfied, it should either return False or
              raise flags.ValidationError(desired_error_message).
      message: str, error message to be shown to the user if validator's
          condition is not satisfied.
    """
        super(SingleFlagValidator, self).__init__(checker, message)
        self.flag_name = flag_name

    def get_flags_names(self):
        return [self.flag_name]

    def print_flags_with_values(self, flag_values):
        return 'flag --%s=%s' % (self.flag_name, flag_values[self.flag_name].value)

    def _get_input_to_checker_function(self, flag_values):
        """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      object, the input to be given to checker.
    """
        return flag_values[self.flag_name].value