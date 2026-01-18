import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def add_diff(self, diff, version_id, parent_ids):
    with open(self._filename + '.mpknit', 'ab') as outfile:
        outfile.seek(0, 2)
        start = outfile.tell()
        with gzip.GzipFile(None, mode='ab', fileobj=outfile) as zipfile:
            zipfile.writelines(itertools.chain([b'version %s\n' % version_id], diff.to_patch()))
        end = outfile.tell()
    self._diff_offset[version_id] = (start, end - start)
    self._parents[version_id] = parent_ids