from gitdb.test.lib import (
from gitdb.stream import DeltaApplyReader
from gitdb.pack import (
from gitdb.base import (
from gitdb.fun import delta_types
from gitdb.exc import UnsupportedOperation
from gitdb.util import to_bin_sha
import pytest
import os
import tempfile
def _assert_index_file(self, index, version, size):
    assert index.packfile_checksum() != index.indexfile_checksum()
    assert len(index.packfile_checksum()) == 20
    assert len(index.indexfile_checksum()) == 20
    assert index.version() == version
    assert index.size() == size
    assert len(index.offsets()) == size
    for oidx in range(index.size()):
        sha = index.sha(oidx)
        assert oidx == index.sha_to_index(sha)
        entry = index.entry(oidx)
        assert len(entry) == 3
        assert entry[0] == index.offset(oidx)
        assert entry[1] == sha
        assert entry[2] == index.crc(oidx)
        for l in (4, 8, 11, 17, 20):
            assert index.partial_sha_to_index(sha[:l], l * 2) == oidx
    self.assertRaises(ValueError, index.partial_sha_to_index, '\x00', 2)