import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
def NEWBUF_test_newbuf(self):
    from ctypes import string_at
    from pygame.tests.test_utils import buftools
    Exporter = buftools.Exporter
    Importer = buftools.Importer
    exp = Exporter((10,), 'B', readonly=True)
    b = BufferProxy(exp)
    self.assertEqual(b.length, exp.len)
    self.assertEqual(b.raw, string_at(exp.buf, exp.len))
    d = b.__array_interface__
    try:
        self.assertEqual(d['typestr'], '|u1')
        self.assertEqual(d['shape'], exp.shape)
        self.assertEqual(d['strides'], exp.strides)
        self.assertEqual(d['data'], (exp.buf, True))
    finally:
        d = None
    exp = Exporter((3,), '=h')
    b = BufferProxy(exp)
    self.assertEqual(b.length, exp.len)
    self.assertEqual(b.raw, string_at(exp.buf, exp.len))
    d = b.__array_interface__
    try:
        lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        f = f'{('<' if lil_endian else '>')}i{exp.itemsize}'
        self.assertEqual(d['typestr'], f)
        self.assertEqual(d['shape'], exp.shape)
        self.assertEqual(d['strides'], exp.strides)
        self.assertEqual(d['data'], (exp.buf, False))
    finally:
        d = None
    exp = Exporter((10, 2), '=i')
    b = BufferProxy(exp)
    imp = Importer(b, buftools.PyBUF_RECORDS)
    self.assertTrue(imp.obj is b)
    self.assertEqual(imp.buf, exp.buf)
    self.assertEqual(imp.ndim, exp.ndim)
    self.assertEqual(imp.format, exp.format)
    self.assertEqual(imp.readonly, exp.readonly)
    self.assertEqual(imp.itemsize, exp.itemsize)
    self.assertEqual(imp.len, exp.len)
    self.assertEqual(imp.shape, exp.shape)
    self.assertEqual(imp.strides, exp.strides)
    self.assertTrue(imp.suboffsets is None)
    d = {'typestr': '|u1', 'shape': (10,), 'strides': (1,), 'data': (9, True)}
    b = BufferProxy(d)
    imp = Importer(b, buftools.PyBUF_SIMPLE)
    self.assertTrue(imp.obj is b)
    self.assertEqual(imp.buf, 9)
    self.assertEqual(imp.len, 10)
    self.assertEqual(imp.format, None)
    self.assertEqual(imp.itemsize, 1)
    self.assertEqual(imp.ndim, 0)
    self.assertTrue(imp.readonly)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)