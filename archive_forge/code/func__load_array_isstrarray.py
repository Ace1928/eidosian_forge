import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _load_array_isstrarray(self, a):
    a = a[1:-1].strip()
    if a != '' and (a[0] == '"' or a[0] == "'"):
        return True
    return False