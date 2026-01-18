import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class Removals(Deb822):
    """Represent an ftp-master removals.822 file

    Removal of packages from the archive are recorded by ftp-masters.
    See https://ftp-master.debian.org/#removed

    Note: this API is experimental and backwards-incompatible changes might be
    required in the future. Please use it and help us improve it!
    """
    __sources_line_re = re.compile('\\s*(?P<package>.+?)_(?P<version>[^\\s]+)\\s*')
    __binaries_line_re = re.compile('\\s*(?P<package>.+?)_(?P<version>[^\\s]+)\\s+\\[(?P<archs>.+)\\]')

    def __init__(self, *args, **kwargs):
        super(Removals, self).__init__(*args, **kwargs)
        self._sources = None
        self._binaries = None

    @property
    def date(self):
        """ a datetime object for the removal action """
        timearray = email.utils.parsedate_tz(self['date'])
        if timearray is None:
            raise ValueError('No date specified')
        ts = email.utils.mktime_tz(timearray)
        return datetime.datetime.fromtimestamp(ts)

    @property
    def bug(self):
        """ list of bug numbers that had requested the package removal

        The bug numbers are returned as integers.

        Note: there is normally only one entry in this list but there may be
        more than one.
        """
        if 'bug' not in self:
            return []
        return [int(b) for b in self['bug'].split(',')]

    @property
    def also_wnpp(self):
        """ list of WNPP bug numbers closed by the removal

        The bug numbers are returned as integers.
        """
        if 'also-wnpp' not in self:
            return []
        return [int(b) for b in self['also-wnpp'].split(' ')]

    @property
    def also_bugs(self):
        """ list of bug numbers in the package closed by the removal

        The bug numbers are returned as integers.

        Removal of a package implicitly also closes all bugs associated with
        the package.
        """
        if 'also-bugs' not in self:
            return []
        return [int(b) for b in self['also-bugs'].split(' ')]

    @property
    def sources(self):
        """ list of source packages that were removed

        A list of dicts is returned, each dict has the form::

            {
                'source': 'some-package-name',
                'version': '1.2.3-1'
            }

        Note: There may be no source packages removed at all if the removal is
        only of a binary package. An empty list is returned in that case.
        """
        if self._sources is not None:
            return self._sources
        s = []
        if 'sources' in self:
            for line in self['sources'].splitlines():
                matches = self.__sources_line_re.match(line)
                if matches:
                    s.append({'source': matches.group('package'), 'version': matches.group('version')})
        self._sources = s
        return s

    @property
    def binaries(self):
        """ list of binary packages that were removed

        A list of dicts is returned, each dict has the form::

            {
                'package': 'some-package-name',
                'version': '1.2.3-1',
                'architectures': set(['i386', 'amd64'])
            }
        """
        if self._binaries is not None:
            return self._binaries
        b = []
        if 'binaries' in self:
            for line in self['binaries'].splitlines():
                matches = self.__binaries_line_re.match(line)
                if matches:
                    b.append({'package': matches.group('package'), 'version': matches.group('version'), 'architectures': set(matches.group('archs').split(', '))})
        self._binaries = b
        return b