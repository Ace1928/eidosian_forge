import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
def _base_validation(self, image_info):
    self.assertEqual(image_info.image, self.image_name)
    self.assertEqual(image_info.file_format, self.file_format)
    self.assertEqual(image_info.virtual_size, self.exp_virtual_size)
    self.assertEqual(image_info.disk_size, self.exp_disk_size)
    if self.snapshot_count is not None:
        self.assertEqual(len(image_info.snapshots), self.snapshot_count)