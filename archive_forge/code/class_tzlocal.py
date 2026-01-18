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
class tzlocal(_tzinfo):
    """
    A :class:`tzinfo` subclass built around the ``time`` timezone functions.
    """

    def __init__(self):
        super(tzlocal, self).__init__()
        self._std_offset = datetime.timedelta(seconds=-time.timezone)
        if time.daylight:
            self._dst_offset = datetime.timedelta(seconds=-time.altzone)
        else:
            self._dst_offset = self._std_offset
        self._dst_saved = self._dst_offset - self._std_offset
        self._hasdst = bool(self._dst_saved)
        self._tznames = tuple(time.tzname)

    def utcoffset(self, dt):
        if dt is None and self._hasdst:
            return None
        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if dt is None and self._hasdst:
            return None
        if self._isdst(dt):
            return self._dst_offset - self._std_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._tznames[self._isdst(dt)]

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        naive_dst = self._naive_is_dst(dt)
        return not naive_dst and naive_dst != self._naive_is_dst(dt - self._dst_saved)

    def _naive_is_dst(self, dt):
        timestamp = _datetime_to_timestamp(dt)
        return time.localtime(timestamp + time.timezone).tm_isdst

    def _isdst(self, dt, fold_naive=True):
        if not self._hasdst:
            return False
        dstval = self._naive_is_dst(dt)
        fold = getattr(dt, 'fold', None)
        if self.is_ambiguous(dt):
            if fold is not None:
                return not self._fold(dt)
            else:
                return True
        return dstval

    def __eq__(self, other):
        if isinstance(other, tzlocal):
            return self._std_offset == other._std_offset and self._dst_offset == other._dst_offset
        elif isinstance(other, tzutc):
            return not self._hasdst and self._tznames[0] in {'UTC', 'GMT'} and (self._std_offset == ZERO)
        elif isinstance(other, tzoffset):
            return not self._hasdst and self._tznames[0] == other._name and (self._std_offset == other._offset)
        else:
            return NotImplemented
    __hash__ = None

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s()' % self.__class__.__name__
    __reduce__ = object.__reduce__