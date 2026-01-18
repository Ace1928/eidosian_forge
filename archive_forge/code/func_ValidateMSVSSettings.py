import re
import sys
def ValidateMSVSSettings(settings, stderr=sys.stderr):
    """Validates that the names of the settings are valid for MSVS.

  Args:
      settings: A dictionary.  The key is the tool name.  The values are
          themselves dictionaries of settings and their values.
      stderr: The stream receiving the error messages.
  """
    _ValidateSettings(_msvs_validators, settings, stderr)