import csv
import datetime
import os
class DistroRelease(object):
    """Represents a distributions release"""

    def __init__(self, version, codename, series, created=None, release=None, eol=None, eol_esm=None, eol_lts=None, eol_elts=None, eol_server=None):
        self.version = version
        self.codename = codename
        self.series = series
        self.created = created
        self.release = release
        self.eol = eol
        self.eol_lts = eol_lts
        self.eol_elts = eol_elts
        self.eol_esm = eol_esm
        self.eol_server = eol_server

    def is_supported(self, date):
        """Check whether this release is supported on the given date."""
        return date >= self.created and (self.eol is None or date <= self.eol or (self.eol_server is not None and date <= self.eol_server))