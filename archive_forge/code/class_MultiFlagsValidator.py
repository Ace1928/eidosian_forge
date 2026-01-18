from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.flags import _exceptions
class MultiFlagsValidator(Validator):
    """Validator behind register_multi_flags_validator method.

  Validates that flag values pass their common checker function. The checker
  function takes flag values and returns True (if values look fine) or,
  if values are not valid, either returns False or raises an Exception.
  """

    def __init__(self, flag_names, checker, message):
        """Constructor.

    Args:
      flag_names: [str], containing names of the flags used by checker.
      checker: function to verify the validator.
          input  - dict, with keys() being flag_names, and value for each
              key being the value of the corresponding flag (string, boolean,
              etc).
          output - bool, True if validator constraint is satisfied.
              If constraint is not satisfied, it should either return False or
              raise flags.ValidationError(desired_error_message).
      message: str, error message to be shown to the user if validator's
          condition is not satisfied
    """
        super(MultiFlagsValidator, self).__init__(checker, message)
        self.flag_names = flag_names

    def _get_input_to_checker_function(self, flag_values):
        """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      dict, with keys() being self.lag_names, and value for each key
      being the value of the corresponding flag (string, boolean, etc).
    """
        return dict(([key, flag_values[key].value] for key in self.flag_names))

    def print_flags_with_values(self, flag_values):
        prefix = 'flags '
        flags_with_values = []
        for key in self.flag_names:
            flags_with_values.append('%s=%s' % (key, flag_values[key].value))
        return prefix + ', '.join(flags_with_values)

    def get_flags_names(self):
        return self.flag_names