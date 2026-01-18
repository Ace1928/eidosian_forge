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
@six.add_metaclass(_TzStrFactory)
class tzstr(tzrange):
    """
    ``tzstr`` objects are time zone objects specified by a time-zone string as
    it would be passed to a ``TZ`` variable on POSIX-style systems (see
    the `GNU C Library: TZ Variable`_ for more details).

    There is one notable exception, which is that POSIX-style time zones use an
    inverted offset format, so normally ``GMT+3`` would be parsed as an offset
    3 hours *behind* GMT. The ``tzstr`` time zone object will parse this as an
    offset 3 hours *ahead* of GMT. If you would like to maintain the POSIX
    behavior, pass a ``True`` value to ``posix_offset``.

    The :class:`tzrange` object provides the same functionality, but is
    specified using :class:`relativedelta.relativedelta` objects. rather than
    strings.

    :param s:
        A time zone string in ``TZ`` variable format. This can be a
        :class:`bytes` (2.x: :class:`str`), :class:`str` (2.x:
        :class:`unicode`) or a stream emitting unicode characters
        (e.g. :class:`StringIO`).

    :param posix_offset:
        Optional. If set to ``True``, interpret strings such as ``GMT+3`` or
        ``UTC+3`` as being 3 hours *behind* UTC rather than ahead, per the
        POSIX standard.

    .. caution::

        Prior to version 2.7.0, this function also supported time zones
        in the format:

            * ``EST5EDT,4,0,6,7200,10,0,26,7200,3600``
            * ``EST5EDT,4,1,0,7200,10,-1,0,7200,3600``

        This format is non-standard and has been deprecated; this function
        will raise a :class:`DeprecatedTZFormatWarning` until
        support is removed in a future version.

    .. _`GNU C Library: TZ Variable`:
        https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html
    """

    def __init__(self, s, posix_offset=False):
        global parser
        from dateutil.parser import _parser as parser
        self._s = s
        res = parser._parsetz(s)
        if res is None or res.any_unused_tokens:
            raise ValueError('unknown string format')
        if res.stdabbr in ('GMT', 'UTC') and (not posix_offset):
            res.stdoffset *= -1
        tzrange.__init__(self, res.stdabbr, res.stdoffset, res.dstabbr, res.dstoffset, start=False, end=False)
        if not res.dstabbr:
            self._start_delta = None
            self._end_delta = None
        else:
            self._start_delta = self._delta(res.start)
            if self._start_delta:
                self._end_delta = self._delta(res.end, isend=1)
        self.hasdst = bool(self._start_delta)

    def _delta(self, x, isend=0):
        from dateutil import relativedelta
        kwargs = {}
        if x.month is not None:
            kwargs['month'] = x.month
            if x.weekday is not None:
                kwargs['weekday'] = relativedelta.weekday(x.weekday, x.week)
                if x.week > 0:
                    kwargs['day'] = 1
                else:
                    kwargs['day'] = 31
            elif x.day:
                kwargs['day'] = x.day
        elif x.yday is not None:
            kwargs['yearday'] = x.yday
        elif x.jyday is not None:
            kwargs['nlyearday'] = x.jyday
        if not kwargs:
            if not isend:
                kwargs['month'] = 4
                kwargs['day'] = 1
                kwargs['weekday'] = relativedelta.SU(+1)
            else:
                kwargs['month'] = 10
                kwargs['day'] = 31
                kwargs['weekday'] = relativedelta.SU(-1)
        if x.time is not None:
            kwargs['seconds'] = x.time
        else:
            kwargs['seconds'] = 7200
        if isend:
            delta = self._dst_offset - self._std_offset
            kwargs['seconds'] -= delta.seconds + delta.days * 86400
        return relativedelta.relativedelta(**kwargs)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self._s))