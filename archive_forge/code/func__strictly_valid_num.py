import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _strictly_valid_num(n):
    n = n.strip()
    if not n:
        return False
    if n[0] == '_':
        return False
    if n[-1] == '_':
        return False
    if '_.' in n or '._' in n:
        return False
    if len(n) == 1:
        return True
    if n[0] == '0' and n[1] not in ['.', 'o', 'b', 'x']:
        return False
    if n[0] == '+' or n[0] == '-':
        n = n[1:]
        if len(n) > 1 and n[0] == '0' and (n[1] != '.'):
            return False
    if '__' in n:
        return False
    return True