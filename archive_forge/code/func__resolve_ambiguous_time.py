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
def _resolve_ambiguous_time(self, dt):
    idx = self._find_last_transition(dt)
    _fold = self._fold(dt)
    if idx is None or idx == 0:
        return idx
    idx_offset = int(not _fold and self.is_ambiguous(dt, idx))
    return idx - idx_offset