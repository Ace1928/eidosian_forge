import csv
import datetime
import os
class DebianDistroInfo(DistroInfo):
    """provides information about Debian's distributions"""

    def __init__(self):
        super().__init__('Debian')

    def codename(self, release, date=None, default=None):
        """Map 'unstable', 'testing', etc. to their codenames."""
        if release == 'unstable':
            codename = self.devel(date)
        elif release == 'testing':
            codename = self.testing(date)
        elif release == 'stable':
            codename = self.stable(date)
        elif release == 'oldstable':
            codename = self.old(date)
        else:
            codename = default
        return codename

    def devel(self, date=None, result='codename'):
        """Get latest development distribution based on the given date."""
        if date is None:
            date = self._date
        distros = [x for x in self._avail(date) if x.release is None or (date < x.release and (x.eol is None or date <= x.eol))]
        if len(distros) < 2:
            raise DistroDataOutdated()
        return self._format(result, distros[-2])

    def old(self, date=None, result='codename'):
        """Get old (stable) Debian distribution based on the given date."""
        if date is None:
            date = self._date
        distros = [x for x in self._avail(date) if x.release is not None and date >= x.release]
        if len(distros) < 2:
            raise DistroDataOutdated()
        return self._format(result, distros[-2])

    def supported(self, date=None, result='codename'):
        """Get list of all supported Debian distributions based on the given
        date."""
        if date is None:
            date = self._date
        distros = [self._format(result, x) for x in self._avail(date) if x.eol is None or date <= x.eol]
        return distros

    def lts_supported(self, date=None, result='codename'):
        """Get list of all LTS supported Debian distributions based on the given
        date."""
        if date is None:
            date = self._date
        distros = [self._format(result, x) for x in self._avail(date) if (x.eol is not None and date > x.eol) and (x.eol_lts is not None and date <= x.eol_lts)]
        return distros

    def elts_supported(self, date=None, result='codename'):
        """Get list of all Extended LTS supported Debian distributions based on
        the given date."""
        if date is None:
            date = self._date
        distros = [self._format(result, x) for x in self._avail(date) if (x.eol_lts is not None and date > x.eol_lts) and (x.eol_elts is not None and date <= x.eol_elts)]
        return distros

    def testing(self, date=None, result='codename'):
        """Get latest testing Debian distribution based on the given date."""
        if date is None:
            date = self._date
        distros = [x for x in self._avail(date) if x.release is None and x.version or (x.release is not None and date < x.release and (x.eol is None or date <= x.eol))]
        if not distros:
            raise DistroDataOutdated()
        return self._format(result, distros[-1])

    def valid(self, codename):
        """Check if the given codename is known."""
        return DistroInfo.valid(self, codename) or codename in ['unstable', 'testing', 'stable', 'oldstable']