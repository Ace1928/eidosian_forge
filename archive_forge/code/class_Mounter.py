from __future__ import unicode_literals
import base64
import codecs
import datetime
from email import message_from_file
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile
from . import __version__, DistlibException
from .compat import sysconfig, ZipFile, fsdecode, text_type, filter
from .database import InstalledDistribution
from .metadata import Metadata, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME
from .util import (FileOperator, convert_path, CSVReader, CSVWriter, Cache,
from .version import NormalizedVersion, UnsupportedVersionError
class Mounter(object):

    def __init__(self):
        self.impure_wheels = {}
        self.libs = {}

    def add(self, pathname, extensions):
        self.impure_wheels[pathname] = extensions
        self.libs.update(extensions)

    def remove(self, pathname):
        extensions = self.impure_wheels.pop(pathname)
        for k, v in extensions:
            if k in self.libs:
                del self.libs[k]

    def find_module(self, fullname, path=None):
        if fullname in self.libs:
            result = self
        else:
            result = None
        return result

    def load_module(self, fullname):
        if fullname in sys.modules:
            result = sys.modules[fullname]
        else:
            if fullname not in self.libs:
                raise ImportError('unable to find extension for %s' % fullname)
            result = _load_dynamic(fullname, self.libs[fullname])
            result.__loader__ = self
            parts = fullname.rsplit('.', 1)
            if len(parts) > 1:
                result.__package__ = parts[0]
        return result