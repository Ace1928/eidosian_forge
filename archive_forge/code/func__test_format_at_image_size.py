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
def _test_format_at_image_size(self, format_name, image_size, subformat=None):
    img = self._create_img(format_name, image_size, subformat=subformat)
    virtual_size = get_size_from_qemu_img(img)
    for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
        fmt = self._test_format_at_block_size(format_name, img, block_size)
        self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
        self.assertEqual(virtual_size, fmt.virtual_size, 'Failed to calculate size for %s at size %i block %i' % (format_name, image_size, block_size))
        memory = sum(fmt.context_info.values())
        self.assertLess(memory, 512 * units.Ki, 'Format used more than 512KiB of memory: %s' % fmt.context_info)