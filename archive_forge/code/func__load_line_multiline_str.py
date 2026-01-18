import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _load_line_multiline_str(self, p):
    poffset = 0
    if len(p) < 3:
        return (-1, poffset)
    if p[0] == '[' and (p.strip()[-1] != ']' and self._load_array_isstrarray(p)):
        newp = p[1:].strip().split(',')
        while len(newp) > 1 and newp[-1][0] != '"' and (newp[-1][0] != "'"):
            newp = newp[:-2] + [newp[-2] + ',' + newp[-1]]
        newp = newp[-1]
        poffset = len(p) - len(newp)
        p = newp
    if p[0] != '"' and p[0] != "'":
        return (-1, poffset)
    if p[1] != p[0] or p[2] != p[0]:
        return (-1, poffset)
    if len(p) > 5 and p[-1] == p[0] and (p[-2] == p[0]) and (p[-3] == p[0]):
        return (-1, poffset)
    return (len(p) - 1, poffset)