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
class MultiMemoryVersionedFile(BaseVersionedFile):
    """Memory-backed pseudo-versionedfile"""

    def __init__(self, snapshot_interval=25, max_snapshots=None):
        BaseVersionedFile.__init__(self, snapshot_interval, max_snapshots)
        self._diffs = {}

    def add_diff(self, diff, version_id, parent_ids):
        self._diffs[version_id] = diff
        self._parents[version_id] = parent_ids

    def get_diff(self, version_id):
        try:
            return self._diffs[version_id]
        except KeyError:
            raise errors.RevisionNotPresent(version_id, self)

    def destroy(self):
        self._diffs = {}