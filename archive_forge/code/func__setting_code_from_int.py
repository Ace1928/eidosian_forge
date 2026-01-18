import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
def _setting_code_from_int(code):
    """
    Given an integer setting code, returns either one of :class:`SettingCodes
    <h2.settings.SettingCodes>` or, if not present in the known set of codes,
    returns the integer directly.
    """
    try:
        return SettingCodes(code)
    except ValueError:
        return code