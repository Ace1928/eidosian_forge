from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _unify_values(self, section, vars):
    """Create a sequence of lookups with 'vars' taking priority over
        the 'section' which takes priority over the DEFAULTSECT.

        """
    sectiondict = {}
    try:
        sectiondict = self._sections[section]
    except KeyError:
        if section != self.default_section:
            raise NoSectionError(section) from None
    vardict = {}
    if vars:
        for key, value in vars.items():
            if value is not None:
                value = str(value)
            vardict[self.optionxform(key)] = value
    return _ChainMap(vardict, sectiondict, self._defaults)