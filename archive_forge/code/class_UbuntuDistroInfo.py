import csv
import datetime
import os
class UbuntuDistroInfo(DistroInfo):
    """provides information about Ubuntu's distributions"""

    def __init__(self):
        super().__init__('Ubuntu')

    def lts(self, date=None, result='codename'):
        """Get latest long term support (LTS) Ubuntu distribution based on the
        given date."""
        if date is None:
            date = self._date
        distros = [x for x in self._releases if x.version.find('LTS') >= 0 and x.release <= date <= x.eol]
        if not distros:
            raise DistroDataOutdated()
        return self._format(result, distros[-1])

    def is_lts(self, codename):
        """Is codename an LTS release?"""
        distros = [x for x in self._releases if x.series == codename]
        if not distros:
            return False
        return 'LTS' in distros[0].version

    def supported(self, date=None, result='codename'):
        """Get list of all supported Ubuntu distributions based on the given
        date."""
        if date is None:
            date = self._date
        distros = [self._format(result, x) for x in self._avail(date) if date <= x.eol or (x.eol_server is not None and date <= x.eol_server)]
        return distros

    def supported_esm(self, date=None, result='codename'):
        """Get list of all ESM supported Ubuntu distributions based on the
        given date."""
        if date is None:
            date = self._date
        distros = [self._format(result, x) for x in self._avail(date) if x.eol_esm is not None and date <= x.eol_esm]
        return distros