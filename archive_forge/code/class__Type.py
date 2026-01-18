import re
import sys
class _Type:
    """Type of settings (Base class)."""

    def ValidateMSVS(self, value):
        """Verifies that the value is legal for MSVS.

    Args:
      value: the value to check for this type.

    Raises:
      ValueError if value is not valid for MSVS.
    """

    def ValidateMSBuild(self, value):
        """Verifies that the value is legal for MSBuild.

    Args:
      value: the value to check for this type.

    Raises:
      ValueError if value is not valid for MSBuild.
    """

    def ConvertToMSBuild(self, value):
        """Returns the MSBuild equivalent of the MSVS value given.

    Args:
      value: the MSVS value to convert.

    Returns:
      the MSBuild equivalent.

    Raises:
      ValueError if value is not valid.
    """
        return value