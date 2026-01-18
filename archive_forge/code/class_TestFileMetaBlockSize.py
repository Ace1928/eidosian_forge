import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
class TestFileMetaBlockSize(TestCase):
    """
        Feature: The meta block size can be manipulated, changing how metadata
        is aggregated and the offset of the first dataset.
    """

    def test_file_create_with_meta_block_size_4096(self):
        meta_block_size = 4096
        with File(self.mktemp(), 'w', meta_block_size=meta_block_size, libver='latest') as f:
            f['test'] = 5
            self.assertEqual(f.meta_block_size, meta_block_size)
            self.assertGreaterEqual(f['test'].id.get_offset(), meta_block_size)

    def test_file_create_with_meta_block_size_512(self):
        meta_block_size = 512
        libver = 'latest'
        with File(self.mktemp(), 'w', meta_block_size=meta_block_size, libver=libver) as f:
            f['test'] = 3
            self.assertEqual(f.meta_block_size, meta_block_size)
            self.assertGreaterEqual(f['test'].id.get_offset(), meta_block_size)
            self.assertLess(f['test'].id.get_offset(), meta_block_size * 2)