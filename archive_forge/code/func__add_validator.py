from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def _add_validator(fv, validator_instance):
    """Register new flags validator to be checked.

  Args:
    fv: flags.FlagValues, the FlagValues instance to add the validator.
    validator_instance: validators.Validator, the validator to add.
  Raises:
    KeyError: Raised when validators work with a non-existing flag.
  """
    for flag_name in validator_instance.get_flags_names():
        fv[flag_name].validators.append(validator_instance)