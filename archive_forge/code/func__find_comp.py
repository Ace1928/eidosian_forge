from relative deltas), local machine timezone, fixed offset timezone, and UTC
import datetime
import logging  # GOOGLE
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict
import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs
from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
from warnings import warn
def _find_comp(self, dt):
    if len(self._comps) == 1:
        return self._comps[0]
    dt = dt.replace(tzinfo=None)
    try:
        with self._cache_lock:
            return self._cachecomp[self._cachedate.index((dt, self._fold(dt)))]
    except ValueError:
        pass
    lastcompdt = None
    lastcomp = None
    for comp in self._comps:
        compdt = self._find_compdt(comp, dt)
        if compdt and (not lastcompdt or lastcompdt < compdt):
            lastcompdt = compdt
            lastcomp = comp
    if not lastcomp:
        for comp in self._comps:
            if not comp.isdst:
                lastcomp = comp
                break
        else:
            lastcomp = comp[0]
    with self._cache_lock:
        self._cachedate.insert(0, (dt, self._fold(dt)))
        self._cachecomp.insert(0, lastcomp)
        if len(self._cachedate) > 10:
            self._cachedate.pop()
            self._cachecomp.pop()
    return lastcomp