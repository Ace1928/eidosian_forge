import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
class tzwinlocal(tzwinbase):
    """
    Class representing the local time zone information in the Windows registry

    While :class:`dateutil.tz.tzlocal` makes system calls (via the :mod:`time`
    module) to retrieve time zone information, ``tzwinlocal`` retrieves the
    rules directly from the Windows registry and creates an object like
    :class:`dateutil.tz.tzwin`.

    Because Windows does not have an equivalent of :func:`time.tzset`, on
    Windows, :class:`dateutil.tz.tzlocal` instances will always reflect the
    time zone settings *at the time that the process was started*, meaning
    changes to the machine's time zone settings during the run of a program
    on Windows will **not** be reflected by :class:`dateutil.tz.tzlocal`.
    Because ``tzwinlocal`` reads the registry directly, it is unaffected by
    this issue.
    """

    def __init__(self):
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZLOCALKEYNAME) as tzlocalkey:
                keydict = valuestodict(tzlocalkey)
            self._std_abbr = keydict['StandardName']
            self._dst_abbr = keydict['DaylightName']
            try:
                tzkeyname = text_type('{kn}\\{sn}').format(kn=TZKEYNAME, sn=self._std_abbr)
                with winreg.OpenKey(handle, tzkeyname) as tzkey:
                    _keydict = valuestodict(tzkey)
                    self._display = _keydict['Display']
            except OSError:
                self._display = None
        stdoffset = -keydict['Bias'] - keydict['StandardBias']
        dstoffset = stdoffset - keydict['DaylightBias']
        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)
        tup = struct.unpack('=8h', keydict['StandardStart'])
        self._stdmonth, self._stdweeknumber, self._stdhour, self._stdminute = tup[1:5]
        self._stddayofweek = tup[7]
        tup = struct.unpack('=8h', keydict['DaylightStart'])
        self._dstmonth, self._dstweeknumber, self._dsthour, self._dstminute = tup[1:5]
        self._dstdayofweek = tup[7]
        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return 'tzwinlocal()'

    def __str__(self):
        return 'tzwinlocal(%s)' % repr(self._std_abbr)

    def __reduce__(self):
        return (self.__class__, ())