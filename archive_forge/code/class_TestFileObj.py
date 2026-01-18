import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
class TestFileObj(TestCase):

    def check_write(self, fileobj):
        f = h5py.File(fileobj, 'w')
        self.assertEqual(f.driver, 'fileobj')
        self.assertEqual(f.filename, repr(fileobj))
        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        f.close()

    def check_read(self, fileobj):
        f = h5py.File(fileobj, 'r')
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        self.assertRaises(Exception, f.create_dataset, 'another.test', data=list(range(3)))
        f.close()

    def test_BytesIO(self):
        with io.BytesIO() as fileobj:
            self.assertEqual(len(fileobj.getvalue()), 0)
            self.check_write(fileobj)
            self.assertGreater(len(fileobj.getvalue()), 0)
            self.check_read(fileobj)

    def test_file(self):
        fname = self.mktemp()
        try:
            with open(fname, 'wb+') as fileobj:
                self.assertEqual(os.path.getsize(fname), 0)
                self.check_write(fileobj)
                self.assertGreater(os.path.getsize(fname), 0)
                self.check_read(fileobj)
            with open(fname, 'rb') as fileobj:
                self.check_read(fileobj)
        finally:
            os.remove(fname)

    def test_TemporaryFile(self):
        fileobj = tempfile.NamedTemporaryFile()
        fname = fileobj.name
        f = h5py.File(fileobj, 'w')
        del fileobj
        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        self.assertTrue(os.path.isfile(fname))
        f.close()
        self.assertFalse(os.path.isfile(fname))

    def test_exception_open(self):
        self.assertRaises(Exception, h5py.File, None, driver='fileobj', mode='x')
        self.assertRaises(Exception, h5py.File, 'rogue', driver='fileobj', mode='x')
        self.assertRaises(Exception, h5py.File, self, driver='fileobj', mode='x')

    def test_exception_read(self):

        class BrokenBytesIO(io.BytesIO):

            def readinto(self, b):
                raise Exception('I am broken')
        f = h5py.File(BrokenBytesIO(), 'w')
        f.create_dataset('test', data=list(range(12)))
        self.assertRaises(Exception, list, f['test'])

    def test_exception_write(self):

        class BrokenBytesIO(io.BytesIO):
            allow_write = False

            def write(self, b):
                if self.allow_write:
                    return super().write(b)
                else:
                    raise Exception('I am broken')
        bio = BrokenBytesIO()
        f = h5py.File(bio, 'w')
        try:
            self.assertRaises(Exception, f.create_dataset, 'test', data=list(range(12)))
        finally:
            bio.allow_write = True
            f.close()

    @ut.skip('Incompletely closed files can cause segfaults')
    def test_exception_close(self):
        fileobj = io.BytesIO()
        f = h5py.File(fileobj, 'w')
        fileobj.close()
        self.assertRaises(Exception, f.close)

    def test_exception_writeonly(self):
        fileobj = open(os.path.join(self.tempdir, 'a.h5'), 'wb')
        with self.assertRaises(io.UnsupportedOperation):
            f = h5py.File(fileobj, 'w')
            group = f.create_group('group')
            group.create_dataset('data', data='foo', dtype=h5py.string_dtype())

    def test_method_vanish(self):
        fileobj = io.BytesIO()
        f = h5py.File(fileobj, 'w')
        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f['test'][:]), list(range(12)))
        fileobj.readinto = None
        self.assertRaises(Exception, list, f['test'])