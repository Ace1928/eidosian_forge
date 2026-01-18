from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.compat.version import LooseVersion, Version
class _Numeric:
    """Class to easily allow comparing numbers

    Largely this exists to make comparing an integer and a string on py3
    so that it works like py2.
    """

    def __init__(self, specifier):
        self.specifier = int(specifier)

    def __repr__(self):
        return repr(self.specifier)

    def __eq__(self, other):
        if isinstance(other, _Numeric):
            return self.specifier == other.specifier
        elif isinstance(other, int):
            return self.specifier == other
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, _Numeric):
            return self.specifier < other.specifier
        elif isinstance(other, int):
            return self.specifier < other
        elif isinstance(other, _Alpha):
            return True
        raise ValueError

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)