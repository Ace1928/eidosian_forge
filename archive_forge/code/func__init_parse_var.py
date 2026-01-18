import sys
import re
import os
from configparser import RawConfigParser
def _init_parse_var(self, name, value):
    self._re[name] = re.compile('\\$\\{%s\\}' % name)
    self._re_sub[name] = value