from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import PY3
import datetime
class _UTC(datetime.tzinfo):
    __slots__ = ()

    def utcoffset(self, dt):
        return _ZERO

    def dst(self, dt):
        return _ZERO

    def tzname(self, dt):
        return 'UTC'