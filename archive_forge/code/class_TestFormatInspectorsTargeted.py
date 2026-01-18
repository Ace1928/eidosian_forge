import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
class TestFormatInspectorsTargeted(test_utils.BaseTestCase):

    def _make_vhd_meta(self, guid_raw, item_length):
        data = struct.pack('<8sHH', b'metadata', 0, 1)
        data += b'0' * 20
        data += guid_raw
        data += struct.pack('<III', 256, item_length, 0)
        data += b'0' * 6
        return data

    def test_vhd_table_over_limit(self):
        ins = format_inspector.VHDXInspector()
        meta = format_inspector.CaptureRegion(0, 0)
        desired = b'012345678ABCDEF0'
        meta.data = self._make_vhd_meta(desired, 33 * 2048)
        ins.new_region('metadata', meta)
        new_region = ins._find_meta_entry(ins._guid(desired))
        self.assertEqual(format_inspector.VHDXInspector.VHDX_METADATA_TABLE_MAX_SIZE, new_region.length)

    def test_vhd_table_under_limit(self):
        ins = format_inspector.VHDXInspector()
        meta = format_inspector.CaptureRegion(0, 0)
        desired = b'012345678ABCDEF0'
        meta.data = self._make_vhd_meta(desired, 16 * 2048)
        ins.new_region('metadata', meta)
        new_region = ins._find_meta_entry(ins._guid(desired))
        self.assertEqual(16 * 2048, new_region.length)