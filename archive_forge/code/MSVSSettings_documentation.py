import re
import sys
Returns the MSBuild equivalent of the MSVS value given.

    Args:
      value: the MSVS value to convert.

    Returns:
      the MSBuild equivalent.

    Raises:
      ValueError if value is not valid.
    