import warnings
from warnings import warn
import breezy
class DeprecatedDict(dict):
    """A dictionary that complains when read or written."""
    is_deprecated = True

    def __init__(self, deprecation_version, variable_name, initial_value, advice):
        """Create a dict that warns when read or modified.

        :param deprecation_version: string for the warning format to raise,
            typically from deprecated_in()
        :param initial_value: The contents of the dict
        :param variable_name: This allows better warnings to be printed
        :param advice: String of advice on what callers should do instead
            of using this variable.
        """
        self._deprecation_version = deprecation_version
        self._variable_name = variable_name
        self._advice = advice
        dict.__init__(self, initial_value)
    __len__ = _dict_deprecation_wrapper(dict.__len__)
    __getitem__ = _dict_deprecation_wrapper(dict.__getitem__)
    __setitem__ = _dict_deprecation_wrapper(dict.__setitem__)
    __delitem__ = _dict_deprecation_wrapper(dict.__delitem__)
    keys = _dict_deprecation_wrapper(dict.keys)
    __contains__ = _dict_deprecation_wrapper(dict.__contains__)