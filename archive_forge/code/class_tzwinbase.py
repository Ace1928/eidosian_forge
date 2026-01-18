import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
class tzwinbase(tzrangebase):
    """tzinfo class based on win32's timezones available in the registry."""

    def __init__(self):
        raise NotImplementedError('tzwinbase is an abstract base class')

    def __eq__(self, other):
        if not isinstance(other, tzwinbase):
            return NotImplemented
        return self._std_offset == other._std_offset and self._dst_offset == other._dst_offset and (self._stddayofweek == other._stddayofweek) and (self._dstdayofweek == other._dstdayofweek) and (self._stdweeknumber == other._stdweeknumber) and (self._dstweeknumber == other._dstweeknumber) and (self._stdhour == other._stdhour) and (self._dsthour == other._dsthour) and (self._stdminute == other._stdminute) and (self._dstminute == other._dstminute) and (self._std_abbr == other._std_abbr) and (self._dst_abbr == other._dst_abbr)

    @staticmethod
    def list():
        """Return a list of all time zones known to the system."""
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZKEYNAME) as tzkey:
                result = [winreg.EnumKey(tzkey, i) for i in range(winreg.QueryInfoKey(tzkey)[0])]
        return result

    def display(self):
        """
        Return the display name of the time zone.
        """
        return self._display

    def transitions(self, year):
        """
        For a given year, get the DST on and off transition times, expressed
        always on the standard time side. For zones with no transitions, this
        function returns ``None``.

        :param year:
            The year whose transitions you would like to query.

        :return:
            Returns a :class:`tuple` of :class:`datetime.datetime` objects,
            ``(dston, dstoff)`` for zones with an annual DST transition, or
            ``None`` for fixed offset zones.
        """
        if not self.hasdst:
            return None
        dston = picknthweekday(year, self._dstmonth, self._dstdayofweek, self._dsthour, self._dstminute, self._dstweeknumber)
        dstoff = picknthweekday(year, self._stdmonth, self._stddayofweek, self._stdhour, self._stdminute, self._stdweeknumber)
        dstoff -= self._dst_base_offset
        return (dston, dstoff)

    def _get_hasdst(self):
        return self._dstmonth != 0

    @property
    def _dst_base_offset(self):
        return self._dst_base_offset_