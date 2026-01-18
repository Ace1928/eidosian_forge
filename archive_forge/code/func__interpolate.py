import sys
import re
import os
from configparser import RawConfigParser
def _interpolate(value):
    for k in self._re.keys():
        value = self._re[k].sub(self._re_sub[k], value)
    return value