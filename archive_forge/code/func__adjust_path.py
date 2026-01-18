from __future__ import unicode_literals
import bisect
import io
import logging
import os
import pkgutil
import sys
import types
import zipimport
from . import DistlibException
from .util import cached_property, get_cache_base, Cache
def _adjust_path(self, path):
    return path