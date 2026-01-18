from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def _get_records(self):
    """
        Get the list of installed files for the distribution
        :return: A list of tuples of path, hash and size. Note that hash and
                 size might be ``None`` for some entries. The path is exactly
                 as stored in the file (which is as in PEP 376).
        """
    results = []
    r = self.get_distinfo_resource('RECORD')
    with contextlib.closing(r.as_stream()) as stream:
        with CSVReader(stream=stream) as record_reader:
            for row in record_reader:
                missing = [None for i in range(len(row), 3)]
                path, checksum, size = row + missing
                results.append((path, checksum, size))
    return results