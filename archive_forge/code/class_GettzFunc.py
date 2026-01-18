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
class GettzFunc(object):
    """
        Retrieve a time zone object from a string representation

        This function is intended to retrieve the :py:class:`tzinfo` subclass
        that best represents the time zone that would be used if a POSIX
        `TZ variable`_ were set to the same value.

        If no argument or an empty string is passed to ``gettz``, local time
        is returned:

        .. code-block:: python3

            >>> gettz()
            tzfile('/etc/localtime')

        This function is also the preferred way to map IANA tz database keys
        to :class:`tzfile` objects:

        .. code-block:: python3

            >>> gettz('Pacific/Kiritimati')
            tzfile('/usr/share/zoneinfo/Pacific/Kiritimati')

        On Windows, the standard is extended to include the Windows-specific
        zone names provided by the operating system:

        .. code-block:: python3

            >>> gettz('Egypt Standard Time')
            tzwin('Egypt Standard Time')

        Passing a GNU ``TZ`` style string time zone specification returns a
        :class:`tzstr` object:

        .. code-block:: python3

            >>> gettz('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')
            tzstr('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')

        :param name:
            A time zone name (IANA, or, on Windows, Windows keys), location of
            a ``tzfile(5)`` zoneinfo file or ``TZ`` variable style time zone
            specifier. An empty string, no argument or ``None`` is interpreted
            as local time.

        :return:
            Returns an instance of one of ``dateutil``'s :py:class:`tzinfo`
            subclasses.

        .. versionchanged:: 2.7.0

            After version 2.7.0, any two calls to ``gettz`` using the same
            input strings will return the same object:

            .. code-block:: python3

                >>> tz.gettz('America/Chicago') is tz.gettz('America/Chicago')
                True

            In addition to improving performance, this ensures that
            `"same zone" semantics`_ are used for datetimes in the same zone.


        .. _`TZ variable`:
            https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html

        .. _`"same zone" semantics`:
            https://blog.ganssle.io/articles/2018/02/aware-datetime-arithmetic.html
        """

    def __init__(self):
        self.__instances = weakref.WeakValueDictionary()
        self.__strong_cache_size = 8
        self.__strong_cache = OrderedDict()
        self._cache_lock = _thread.allocate_lock()

    def __call__(self, name=None):
        with self._cache_lock:
            rv = self.__instances.get(name, None)
            if rv is None:
                rv = self.nocache(name=name)
                if not (name is None or isinstance(rv, tzlocal_classes) or rv is None):
                    self.__instances[name] = rv
                else:
                    return rv
            self.__strong_cache[name] = self.__strong_cache.pop(name, rv)
            if len(self.__strong_cache) > self.__strong_cache_size:
                self.__strong_cache.popitem(last=False)
        return rv

    def set_cache_size(self, size):
        with self._cache_lock:
            self.__strong_cache_size = size
            while len(self.__strong_cache) > size:
                self.__strong_cache.popitem(last=False)

    def cache_clear(self):
        with self._cache_lock:
            self.__instances = weakref.WeakValueDictionary()
            self.__strong_cache.clear()

    @staticmethod
    def nocache(name=None):
        """A non-cached version of gettz"""
        tz = None
        if not name:
            try:
                name = os.environ['TZ']
            except KeyError:
                pass
        if name is None or name in ('', ':'):
            for filepath in TZFILES:
                if not os.path.isabs(filepath):
                    filename = filepath
                    for path in TZPATHS:
                        filepath = os.path.join(path, filename)
                        if os.path.isfile(filepath):
                            break
                    else:
                        continue
                if os.path.isfile(filepath):
                    try:
                        tz = tzfile(filepath)
                        break
                    except (IOError, OSError, ValueError):
                        pass
            else:
                tz = tzlocal()
        else:
            try:
                if name.startswith(':'):
                    name = name[1:]
            except TypeError as e:
                if isinstance(name, bytes):
                    new_msg = 'gettz argument should be str, not bytes'
                    six.raise_from(TypeError(new_msg), e)
                else:
                    raise
            if os.path.isabs(name):
                if os.path.isfile(name):
                    tz = tzfile(name)
                else:
                    tz = None
            else:
                for path in TZPATHS:
                    filepath = os.path.join(path, name)
                    if not os.path.isfile(filepath):
                        filepath = filepath.replace(' ', '_')
                        if not os.path.isfile(filepath):
                            continue
                    try:
                        tz = tzfile(filepath)
                        break
                    except (IOError, OSError, ValueError):
                        pass
                else:
                    tz = None
                    if tzwin is not None:
                        try:
                            tz = tzwin(name)
                        except (WindowsError, UnicodeEncodeError):
                            tz = None
                    if not tz:
                        try:
                            from dateutil.zoneinfo import get_zonefile_instance
                            tz = get_zonefile_instance().get(name)
                        except ImportError:
                            logging.info('dateutil.zoneinfo module not available; using system TZ database.')
                            logging.info('Add "//third_party/py/dateutil:zoneinfo" to your deps if you need your own copy.')
                    if not tz:
                        for c in name:
                            if c in '0123456789':
                                try:
                                    tz = tzstr(name)
                                except ValueError:
                                    pass
                                break
                        else:
                            if name in ('GMT', 'UTC'):
                                tz = UTC
                            elif name in time.tzname:
                                tz = tzlocal()
        return tz