import sys
import unittest
import platform
import pygame
def NEWBUF_assertSame(self, proxy, exp):
    buftools = self.buftools
    Importer = buftools.Importer
    self.assertEqual(proxy.length, exp.len)
    imp = Importer(proxy, buftools.PyBUF_RECORDS_RO)
    self.assertEqual(imp.readonly, exp.readonly)
    self.assertEqual(imp.format, exp.format)
    self.assertEqual(imp.itemsize, exp.itemsize)
    self.assertEqual(imp.ndim, exp.ndim)
    self.assertEqual(imp.shape, exp.shape)
    self.assertEqual(imp.strides, exp.strides)
    self.assertTrue(imp.suboffsets is None)