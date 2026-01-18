import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype
from h5py._hl.files import direct_vfd
from .common import ut, TestCase
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
class TestFileID(TestCase):

    def test_descriptor_core(self):
        with File('TestFileID.test_descriptor_core', driver='core', backing_store=False, mode='x') as f:
            assert isinstance(f.id.get_vfd_handle(), int)

    def test_descriptor_sec2(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.test_descriptor_sec2')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, driver='sec2', mode='x') as f:
                descriptor = f.id.get_vfd_handle()
                self.assertNotEqual(descriptor, 0)
                os.fsync(descriptor)
        finally:
            shutil.rmtree(dn_tmp)

    @ut.skipUnless(direct_vfd, 'DIRECT driver is supported on Linux if hdf5 is built with the appriorate flags.')
    def test_descriptor_direct(self):
        dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.test_descriptor_direct')
        fn_h5 = os.path.join(dn_tmp, 'test.h5')
        try:
            with File(fn_h5, driver='direct', mode='x') as f:
                descriptor = f.id.get_vfd_handle()
                self.assertNotEqual(descriptor, 0)
                os.fsync(descriptor)
        finally:
            shutil.rmtree(dn_tmp)