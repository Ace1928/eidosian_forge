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
class MultiVersionedFile(BaseVersionedFile):
    """Disk-backed pseudo-versionedfile"""

    def __init__(self, filename, snapshot_interval=25, max_snapshots=None):
        BaseVersionedFile.__init__(self, snapshot_interval, max_snapshots)
        self._filename = filename
        self._diff_offset = {}

    def get_diff(self, version_id):
        start, count = self._diff_offset[version_id]
        with open(self._filename + '.mpknit', 'rb') as infile:
            infile.seek(start)
            sio = BytesIO(infile.read(count))
        with gzip.GzipFile(None, mode='rb', fileobj=sio) as zip_file:
            file_version_id = zip_file.readline()
            content = zip_file.read()
            return MultiParent.from_patch(content)

    def add_diff(self, diff, version_id, parent_ids):
        with open(self._filename + '.mpknit', 'ab') as outfile:
            outfile.seek(0, 2)
            start = outfile.tell()
            with gzip.GzipFile(None, mode='ab', fileobj=outfile) as zipfile:
                zipfile.writelines(itertools.chain([b'version %s\n' % version_id], diff.to_patch()))
            end = outfile.tell()
        self._diff_offset[version_id] = (start, end - start)
        self._parents[version_id] = parent_ids

    def destroy(self):
        try:
            os.unlink(self._filename + '.mpknit')
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        try:
            os.unlink(self._filename + '.mpidx')
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def save(self):
        import fastbencode as bencode
        with open(self._filename + '.mpidx', 'wb') as f:
            f.write(bencode.bencode((self._parents, list(self._snapshots), self._diff_offset)))

    def load(self):
        import fastbencode as bencode
        with open(self._filename + '.mpidx', 'rb') as f:
            self._parents, snapshots, self._diff_offset = bencode.bdecode(f.read())
        self._snapshots = set(snapshots)