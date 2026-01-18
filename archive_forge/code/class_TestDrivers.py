import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
class TestDrivers(TestCase):
    """
        Feature: Files can be opened with low-level HDF5 drivers. Does not
        include MPI drivers (see bottom).
    """

    @ut.skipUnless(os.name == 'posix', 'Stdio driver is supported on posix')
    def test_stdio(self):
        """ Stdio driver is supported on posix """
        fid = File(self.mktemp(), 'w', driver='stdio')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'stdio')
        fid.close()
        fid = File(self.mktemp(), 'a', driver='stdio')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'stdio')
        fid.close()

    @ut.skipUnless(direct_vfd, 'DIRECT driver is supported on Linux if hdf5 is built with the appriorate flags.')
    def test_direct(self):
        """ DIRECT driver is supported on Linux"""
        fid = File(self.mktemp(), 'w', driver='direct')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'direct')
        default_fapl = fid.id.get_access_plist().get_fapl_direct()
        fid.close()
        fid = File(self.mktemp(), 'a', driver='direct')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'direct')
        fid.close()
        for alignment, block_size, cbuf_size in [default_fapl, (default_fapl[0], default_fapl[1], 3 * default_fapl[1]), (default_fapl[0] * 2, default_fapl[1], 3 * default_fapl[1]), (default_fapl[0], 2 * default_fapl[1], 6 * default_fapl[1])]:
            with File(self.mktemp(), 'w', driver='direct', alignment=alignment, block_size=block_size, cbuf_size=cbuf_size) as fid:
                actual_fapl = fid.id.get_access_plist().get_fapl_direct()
                actual_alignment = actual_fapl[0]
                actual_block_size = actual_fapl[1]
                actual_cbuf_size = actual_fapl[2]
                assert actual_alignment == alignment
                assert actual_block_size == block_size
                assert actual_cbuf_size == actual_cbuf_size

    @ut.skipUnless(os.name == 'posix', 'Sec2 driver is supported on posix')
    def test_sec2(self):
        """ Sec2 driver is supported on posix """
        fid = File(self.mktemp(), 'w', driver='sec2')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'sec2')
        fid.close()
        fid = File(self.mktemp(), 'a', driver='sec2')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'sec2')
        fid.close()

    def test_core(self):
        """ Core driver is supported (no backing store) """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', backing_store=False)
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'core')
        fid.close()
        self.assertFalse(os.path.exists(fname))
        fid = File(self.mktemp(), 'a', driver='core')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'core')
        fid.close()

    def test_backing(self):
        """ Core driver saves to file when backing store used """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', backing_store=True)
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r')
        assert 'foo' in fid
        fid.close()
        with self.assertRaises(TypeError):
            File(fname, 'w', backing_store=True)

    def test_readonly(self):
        """ Core driver can be used to open existing files """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r', driver='core')
        self.assertTrue(fid)
        assert 'foo' in fid
        with self.assertRaises(ValueError):
            fid.create_group('bar')
        fid.close()

    def test_blocksize(self):
        """ Core driver supports variable block size """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', block_size=1024, backing_store=False)
        self.assertTrue(fid)
        fid.close()

    def test_split(self):
        """ Split stores metadata in a separate file """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='split')
        fid.close()
        self.assertTrue(os.path.exists(fname + '-m.h5'))
        fid = File(fname, 'r', driver='split')
        self.assertTrue(fid)
        fid.close()

    def test_fileobj(self):
        """ Python file object driver is supported """
        tf = tempfile.TemporaryFile()
        fid = File(tf, 'w', driver='fileobj')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'fileobj')
        fid.close()
        with self.assertRaises(ValueError):
            File(tf, 'w', driver='core')