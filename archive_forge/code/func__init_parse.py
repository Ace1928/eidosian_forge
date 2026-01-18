import sys
import re
import os
from configparser import RawConfigParser
def _init_parse(self):
    for k, v in self._raw_data.items():
        self._init_parse_var(k, v)