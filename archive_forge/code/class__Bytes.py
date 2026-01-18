from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
class _Bytes(str):
    """Compat class for displaying bytes on Python 2."""

    def __repr__(self):
        return 'b' + str.__repr__(self)

    def __unicode__(self):
        return self.decode('ascii', 'replace')