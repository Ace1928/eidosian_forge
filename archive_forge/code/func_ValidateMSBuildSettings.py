import re
import sys
def ValidateMSBuildSettings(settings, stderr=sys.stderr):
    """Validates that the names of the settings are valid for MSBuild.

  Args:
      settings: A dictionary.  The key is the tool name.  The values are
          themselves dictionaries of settings and their values.
      stderr: The stream receiving the error messages.
  """
    _ValidateSettings(_msbuild_validators, settings, stderr)