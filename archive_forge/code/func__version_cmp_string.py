import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
@classmethod
def _version_cmp_string(cls, va, vb):
    la = [cls._order(x) for x in va]
    lb = [cls._order(x) for x in vb]
    while la or lb:
        a = 0
        b = 0
        if la:
            a = la.pop(0)
        if lb:
            b = lb.pop(0)
        if a < b:
            return -1
        if a > b:
            return 1
    return 0