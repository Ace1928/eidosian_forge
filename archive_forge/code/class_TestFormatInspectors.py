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
class TestFormatInspectors(test_utils.BaseTestCase):

    def setUp(self):
        super(TestFormatInspectors, self).setUp()
        self._created_files = []

    def tearDown(self):
        super(TestFormatInspectors, self).tearDown()
        for fn in self._created_files:
            try:
                os.remove(fn)
            except Exception:
                pass

    def _create_img(self, fmt, size, subformat=None):
        if fmt == 'vhd':
            fmt = 'vpc'
        opt = ''
        prefix = 'glance-unittest-formatinspector-'
        if subformat:
            opt = ' -o subformat=%s' % subformat
            prefix += subformat + '-'
        fn = tempfile.mktemp(prefix=prefix, suffix='.%s' % fmt)
        self._created_files.append(fn)
        subprocess.check_output('qemu-img create -f %s %s %s %i' % (fmt, opt, fn, size), shell=True)
        return fn

    def _create_allocated_vmdk(self, size_mb, subformat=None):
        if subformat is None:
            subformat = 'monolithicSparse'
        prefix = 'glance-unittest-formatinspector-%s-' % subformat
        fn = tempfile.mktemp(prefix=prefix, suffix='.vmdk')
        self._created_files.append(fn)
        raw = tempfile.mktemp(prefix=prefix, suffix='.raw')
        self._created_files.append(raw)
        subprocess.check_output('dd if=/dev/urandom of=%s bs=1M count=%i' % (raw, size_mb), shell=True)
        subprocess.check_output('qemu-img convert -f raw -O vmdk -o subformat=%s -S 0 %s %s' % (subformat, raw, fn), shell=True)
        return fn

    def _test_format_at_block_size(self, format_name, img, block_size):
        fmt = format_inspector.get_inspector(format_name)()
        self.assertIsNotNone(fmt, 'Did not get format inspector for %s' % format_name)
        wrapper = format_inspector.InfoWrapper(open(img, 'rb'), fmt)
        while True:
            chunk = wrapper.read(block_size)
            if not chunk:
                break
        wrapper.close()
        return fmt

    def _test_format_at_image_size(self, format_name, image_size, subformat=None):
        img = self._create_img(format_name, image_size, subformat=subformat)
        virtual_size = get_size_from_qemu_img(img)
        for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
            fmt = self._test_format_at_block_size(format_name, img, block_size)
            self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
            self.assertEqual(virtual_size, fmt.virtual_size, 'Failed to calculate size for %s at size %i block %i' % (format_name, image_size, block_size))
            memory = sum(fmt.context_info.values())
            self.assertLess(memory, 512 * units.Ki, 'Format used more than 512KiB of memory: %s' % fmt.context_info)

    def _test_format(self, format_name, subformat=None):
        for image_size in (512, 513, 2057, 7):
            self._test_format_at_image_size(format_name, image_size * units.Mi, subformat=subformat)

    def test_qcow2(self):
        self._test_format('qcow2')

    def test_vhd(self):
        self._test_format('vhd')

    def test_vhdx(self):
        self._test_format('vhdx')

    def test_vmdk(self):
        self._test_format('vmdk')

    def test_vmdk_stream_optimized(self):
        self._test_format('vmdk', 'streamOptimized')

    def _test_vmdk_bad_descriptor_offset(self, subformat=None):
        format_name = 'vmdk'
        image_size = 10 * units.Mi
        descriptorOffsetAddr = 28
        BAD_ADDRESS = 1024
        img = self._create_img(format_name, image_size, subformat=subformat)
        fd = open(img, 'r+b')
        fd.seek(descriptorOffsetAddr)
        fd.write(struct.pack('<Q', BAD_ADDRESS // 512))
        fd.close()
        for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
            fmt = self._test_format_at_block_size(format_name, img, block_size)
            self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
            self.assertEqual(0, fmt.virtual_size, 'Calculated a virtual size for a corrupt %s at size %i block %i' % (format_name, image_size, block_size))

    def test_vmdk_bad_descriptor_offset(self):
        self._test_vmdk_bad_descriptor_offset()

    def test_vmdk_bad_descriptor_offset_stream_optimized(self):
        self._test_vmdk_bad_descriptor_offset(subformat='streamOptimized')

    def _test_vmdk_bad_descriptor_mem_limit(self, subformat=None):
        format_name = 'vmdk'
        image_size = 5 * units.Mi
        virtual_size = 5 * units.Mi
        descriptorOffsetAddr = 28
        descriptorSizeAddr = descriptorOffsetAddr + 8
        twoMBInSectors = (2 << 20) // 512
        img = self._create_allocated_vmdk(image_size // units.Mi, subformat=subformat)
        fd = open(img, 'r+b')
        fd.seek(descriptorSizeAddr)
        fd.write(struct.pack('<Q', twoMBInSectors))
        fd.close()
        for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
            fmt = self._test_format_at_block_size(format_name, img, block_size)
            self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
            self.assertEqual(virtual_size, fmt.virtual_size, 'Failed to calculate size for %s at size %i block %i' % (format_name, image_size, block_size))
            memory = sum(fmt.context_info.values())
            self.assertLess(memory, 1.5 * units.Mi, 'Format used more than 1.5MiB of memory: %s' % fmt.context_info)

    def test_vmdk_bad_descriptor_mem_limit(self):
        self._test_vmdk_bad_descriptor_mem_limit()

    def test_vmdk_bad_descriptor_mem_limit_stream_optimized(self):
        self._test_vmdk_bad_descriptor_mem_limit(subformat='streamOptimized')

    def test_vdi(self):
        self._test_format('vdi')

    def _test_format_with_invalid_data(self, format_name):
        fmt = format_inspector.get_inspector(format_name)()
        wrapper = format_inspector.InfoWrapper(open(__file__, 'rb'), fmt)
        while True:
            chunk = wrapper.read(32)
            if not chunk:
                break
        wrapper.close()
        self.assertFalse(fmt.format_match)
        self.assertEqual(0, fmt.virtual_size)
        memory = sum(fmt.context_info.values())
        self.assertLess(memory, 512 * units.Ki, 'Format used more than 512KiB of memory: %s' % fmt.context_info)

    def test_qcow2_invalid(self):
        self._test_format_with_invalid_data('qcow2')

    def test_vhd_invalid(self):
        self._test_format_with_invalid_data('vhd')

    def test_vhdx_invalid(self):
        self._test_format_with_invalid_data('vhdx')

    def test_vmdk_invalid(self):
        self._test_format_with_invalid_data('vmdk')

    def test_vdi_invalid(self):
        self._test_format_with_invalid_data('vdi')

    def test_vmdk_invalid_type(self):
        fmt = format_inspector.get_inspector('vmdk')()
        wrapper = format_inspector.InfoWrapper(open(__file__, 'rb'), fmt)
        while True:
            chunk = wrapper.read(32)
            if not chunk:
                break
        wrapper.close()
        fake_rgn = mock.MagicMock()
        fake_rgn.complete = True
        fake_rgn.data = b'foocreateType="someunknownformat"bar'
        with mock.patch.object(fmt, 'has_region', return_value=True):
            with mock.patch.object(fmt, 'region', return_value=fake_rgn):
                self.assertEqual(0, fmt.virtual_size)