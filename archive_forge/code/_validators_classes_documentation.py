from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.flags import _exceptions
Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      dict, with keys() being self.lag_names, and value for each key
      being the value of the corresponding flag (string, boolean, etc).
    