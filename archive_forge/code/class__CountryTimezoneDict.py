import sys
import datetime
import os.path
from pytz.exceptions import AmbiguousTimeError
from pytz.exceptions import InvalidTimeError
from pytz.exceptions import NonExistentTimeError
from pytz.exceptions import UnknownTimeZoneError
from pytz.lazy import LazyDict, LazyList, LazySet  # noqa
from pytz.tzinfo import unpickler, BaseTzInfo
from pytz.tzfile import build_tzinfo
class _CountryTimezoneDict(LazyDict):
    """Map ISO 3166 country code to a list of timezone names commonly used
    in that country.

    iso3166_code is the two letter code used to identify the country.

    >>> def print_list(list_of_strings):
    ...     'We use a helper so doctests work under Python 2.3 -> 3.x'
    ...     for s in list_of_strings:
    ...         print(s)

    >>> print_list(country_timezones['nz'])
    Pacific/Auckland
    Pacific/Chatham
    >>> print_list(country_timezones['ch'])
    Europe/Zurich
    >>> print_list(country_timezones['CH'])
    Europe/Zurich
    >>> print_list(country_timezones[unicode('ch')])
    Europe/Zurich
    >>> print_list(country_timezones['XXX'])
    Traceback (most recent call last):
    ...
    KeyError: 'XXX'

    Previously, this information was exposed as a function rather than a
    dictionary. This is still supported::

    >>> print_list(country_timezones('nz'))
    Pacific/Auckland
    Pacific/Chatham
    """

    def __call__(self, iso3166_code):
        """Backwards compatibility."""
        return self[iso3166_code]

    def _fill(self):
        data = {}
        zone_tab = open_resource('zone.tab')
        try:
            for line in zone_tab:
                line = line.decode('UTF-8')
                if line.startswith('#'):
                    continue
                code, coordinates, zone = line.split(None, 4)[:3]
                if zone not in all_timezones_set:
                    continue
                try:
                    data[code].append(zone)
                except KeyError:
                    data[code] = [zone]
            self.data = data
        finally:
            zone_tab.close()