from contextlib import ExitStack
from functools import wraps, total_ordering
from inspect import getfullargspec as getargspec
import logging
import os
import re
import threading
import warnings
import attr
from rasterio._env import (
from rasterio._version import gdal_version
from rasterio.errors import EnvError, GDALVersionError, RasterioDeprecationWarning
from rasterio.session import Session, DummySession
@attr.s(slots=True)
@total_ordering
class GDALVersion:
    """Convenience class for obtaining GDAL major and minor version components
    and comparing between versions.  This is highly simplistic and assumes a
    very normal numbering scheme for versions and ignores everything except
    the major and minor components."""
    major = attr.ib(default=0, validator=attr.validators.instance_of(int))
    minor = attr.ib(default=0, validator=attr.validators.instance_of(int))

    def __eq__(self, other):
        return (self.major, self.minor) == tuple(other.major, other.minor)

    def __lt__(self, other):
        return (self.major, self.minor) < tuple(other.major, other.minor)

    def __repr__(self):
        return 'GDALVersion(major={0}, minor={1})'.format(self.major, self.minor)

    def __str__(self):
        return '{0}.{1}'.format(self.major, self.minor)

    @classmethod
    def parse(cls, input):
        """
        Parses input tuple or string to GDALVersion. If input is a GDALVersion
        instance, it is returned.

        Parameters
        ----------
        input: tuple of (major, minor), string, or instance of GDALVersion

        Returns
        -------
        GDALVersion instance
        """
        if isinstance(input, cls):
            return input
        if isinstance(input, tuple):
            return cls(*input)
        elif isinstance(input, str):
            match = re.search('^\\d+\\.\\d+', input)
            if not match:
                raise ValueError('value does not appear to be a valid GDAL version number: {}'.format(input))
            major, minor = (int(c) for c in match.group().split('.'))
            return cls(major=major, minor=minor)
        raise TypeError('GDALVersion can only be parsed from a string or tuple')

    @classmethod
    def runtime(cls):
        """Return GDALVersion of current GDAL runtime"""
        return cls.parse(gdal_version())

    def at_least(self, other):
        other = self.__class__.parse(other)
        return self >= other