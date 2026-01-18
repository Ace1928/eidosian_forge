import re
import sys
def _ValidateExclusionSetting(setting, settings, error_msg, stderr=sys.stderr):
    """Verify that 'setting' is valid if it is generated from an exclusion list.

  If the setting appears to be generated from an exclusion list, the root name
  is checked.

  Args:
      setting:   A string that is the setting name to validate
      settings:  A dictionary where the keys are valid settings
      error_msg: The message to emit in the event of error
      stderr:    The stream receiving the error messages.
  """
    unrecognized = True
    m = re.match(_EXCLUDED_SUFFIX_RE, setting)
    if m:
        root_setting = m.group(1)
        unrecognized = root_setting not in settings
    if unrecognized:
        print(error_msg, file=stderr)