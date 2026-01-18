import re
import sys
def _ValidateSettings(validators, settings, stderr):
    """Validates that the settings are valid for MSBuild or MSVS.

  We currently only validate the names of the settings, not their values.

  Args:
      validators: A dictionary of tools and their validators.
      settings: A dictionary.  The key is the tool name.  The values are
          themselves dictionaries of settings and their values.
      stderr: The stream receiving the error messages.
  """
    for tool_name in settings:
        if tool_name in validators:
            tool_validators = validators[tool_name]
            for setting, value in settings[tool_name].items():
                if setting in tool_validators:
                    try:
                        tool_validators[setting](value)
                    except ValueError as e:
                        print(f'Warning: for {tool_name}/{setting}, {e}', file=stderr)
                else:
                    _ValidateExclusionSetting(setting, tool_validators, f'Warning: unrecognized setting {tool_name}/{setting}', stderr)
        else:
            print('Warning: unrecognized tool %s' % tool_name, file=stderr)