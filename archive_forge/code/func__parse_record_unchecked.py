import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _parse_record_unchecked(self, data):
    with gzip.GzipFile(mode='rb', fileobj=BytesIO(data)) as df:
        try:
            record_contents = df.readlines()
        except Exception as e:
            raise KnitCorrupt(self, 'Corrupt compressed record %r, got %s(%s)' % (data, e.__class__.__name__, str(e)))
        header = record_contents.pop(0)
        rec = self._split_header(header)
        last_line = record_contents.pop()
        if len(record_contents) != int(rec[2]):
            raise KnitCorrupt(self, 'incorrect number of lines %s != %s for version {%s} %s' % (len(record_contents), int(rec[2]), rec[1], record_contents))
        if last_line != b'end %s\n' % rec[1]:
            raise KnitCorrupt(self, 'unexpected version end line %r, wanted %r' % (last_line, rec[1]))
    return (rec, record_contents)