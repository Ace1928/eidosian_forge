import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
class InvalidURLJoin(errors.PathError):
    _fmt = 'Invalid URL join request: %(reason)s: %(base)r + %(join_args)r'

    def __init__(self, reason, base, join_args):
        self.reason = reason
        self.base = base
        self.join_args = join_args
        errors.PathError.__init__(self, base, reason)