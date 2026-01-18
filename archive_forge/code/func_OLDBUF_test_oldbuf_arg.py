import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
def OLDBUF_test_oldbuf_arg(self):
    from pygame.bufferproxy import get_segcount, get_read_buffer, get_write_buffer
    content = b'\x01\x00\x00\x02' * 12
    memory = ctypes.create_string_buffer(content)
    memaddr = ctypes.addressof(memory)

    def raise_exception(o):
        raise ValueError('An exception')
    bf = BufferProxy({'shape': (len(content),), 'typestr': '|u1', 'data': (memaddr, False), 'strides': (1,)})
    seglen, segaddr = get_read_buffer(bf, 0)
    self.assertEqual(segaddr, 0)
    self.assertEqual(seglen, 0)
    seglen, segaddr = get_write_buffer(bf, 0)
    self.assertEqual(segaddr, 0)
    self.assertEqual(seglen, 0)
    segcount, buflen = get_segcount(bf)
    self.assertEqual(segcount, 1)
    self.assertEqual(buflen, len(content))
    seglen, segaddr = get_read_buffer(bf, 0)
    self.assertEqual(segaddr, memaddr)
    self.assertEqual(seglen, len(content))
    seglen, segaddr = get_write_buffer(bf, 0)
    self.assertEqual(segaddr, memaddr)
    self.assertEqual(seglen, len(content))
    bf = BufferProxy({'shape': (len(content),), 'typestr': '|u1', 'data': (memaddr, True), 'strides': (1,)})
    segcount, buflen = get_segcount(bf)
    self.assertEqual(segcount, 1)
    self.assertEqual(buflen, len(content))
    seglen, segaddr = get_read_buffer(bf, 0)
    self.assertEqual(segaddr, memaddr)
    self.assertEqual(seglen, len(content))
    self.assertRaises(ValueError, get_write_buffer, bf, 0)
    bf = BufferProxy({'shape': (len(content),), 'typestr': '|u1', 'data': (memaddr, True), 'strides': (1,), 'before': raise_exception})
    segcount, buflen = get_segcount(bf)
    self.assertEqual(segcount, 0)
    self.assertEqual(buflen, 0)
    bf = BufferProxy({'shape': (3, 4), 'typestr': '|u4', 'data': (memaddr, True), 'strides': (12, 4)})
    segcount, buflen = get_segcount(bf)
    self.assertEqual(segcount, 3 * 4)
    self.assertEqual(buflen, 3 * 4 * 4)
    for i in range(0, 4):
        seglen, segaddr = get_read_buffer(bf, i)
        self.assertEqual(segaddr, memaddr + i * 4)
        self.assertEqual(seglen, 4)