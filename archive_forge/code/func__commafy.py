import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def _commafy(s):
    if s.startswith('-'):
        return '-' + _commafy(s[1:])
    elif len(s) <= 3:
        return s
    else:
        return _commafy(s[:-3]) + ',' + _commafy(s[-3:])