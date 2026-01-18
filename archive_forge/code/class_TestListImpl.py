import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
class TestListImpl(TestCase):

    def setUp(self):
        """Bind to the c_helper library and provide the ctypes wrapper.
        """
        list_t = ctypes.c_void_p
        iter_t = ctypes.c_void_p

        def wrap(name, restype, argtypes=()):
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return proto(_helperlib.c_helpers[name])
        self.numba_test_list = wrap('test_list', ctypes.c_int)
        self.numba_list_new = wrap('list_new', ctypes.c_int, [ctypes.POINTER(list_t), ctypes.c_ssize_t, ctypes.c_ssize_t])
        self.numba_list_free = wrap('list_free', None, [list_t])
        self.numba_list_length = wrap('list_length', ctypes.c_int, [list_t])
        self.numba_list_allocated = wrap('list_allocated', ctypes.c_int, [list_t])
        self.numba_list_is_mutable = wrap('list_is_mutable', ctypes.c_int, [list_t])
        self.numba_list_set_is_mutable = wrap('list_set_is_mutable', None, [list_t, ctypes.c_int])
        self.numba_list_setitem = wrap('list_setitem', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_char_p])
        self.numba_list_append = wrap('list_append', ctypes.c_int, [list_t, ctypes.c_char_p])
        self.numba_list_getitem = wrap('list_getitem', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_char_p])
        self.numba_list_delitem = wrap('list_delitem', ctypes.c_int, [list_t, ctypes.c_ssize_t])
        self.numba_list_delete_slice = wrap('list_delete_slice', ctypes.c_int, [list_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t])
        self.numba_list_iter_sizeof = wrap('list_iter_sizeof', ctypes.c_size_t)
        self.numba_list_iter = wrap('list_iter', None, [iter_t, list_t])
        self.numba_list_iter_next = wrap('list_iter_next', ctypes.c_int, [iter_t, ctypes.POINTER(ctypes.c_void_p)])

    def test_simple_c_test(self):
        ret = self.numba_test_list()
        self.assertEqual(ret, 0)

    def test_length(self):
        l = List(self, 8, 0)
        self.assertEqual(len(l), 0)

    def test_allocation(self):
        for i in range(16):
            l = List(self, 8, i)
            self.assertEqual(len(l), 0)
            self.assertEqual(l.allocated, i)

    def test_append_get_string(self):
        l = List(self, 8, 1)
        l.append(b'abcdefgh')
        self.assertEqual(len(l), 1)
        r = l[0]
        self.assertEqual(r, b'abcdefgh')

    def test_append_get_int(self):
        l = List(self, 8, 1)
        l.append(struct.pack('q', 1))
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)

    def test_append_get_string_realloc(self):
        l = List(self, 8, 1)
        l.append(b'abcdefgh')
        self.assertEqual(len(l), 1)
        l.append(b'hijklmno')
        self.assertEqual(len(l), 2)
        r = l[1]
        self.assertEqual(r, b'hijklmno')

    def test_set_item_getitem_index_error(self):
        l = List(self, 8, 0)
        with self.assertRaises(IndexError):
            l[0]
        with self.assertRaises(IndexError):
            l[0] = b'abcdefgh'

    def test_iter(self):
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        received = []
        for j in l:
            received.append(j)
        self.assertEqual(values, received)

    def test_pop(self):
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        received = l.pop()
        self.assertEqual(b'h', received)
        self.assertEqual(len(l), 7)
        received = [j for j in l]
        self.assertEqual(received, values[:-1])
        received = l.pop(0)
        self.assertEqual(b'a', received)
        self.assertEqual(len(l), 6)
        received = l.pop(2)
        self.assertEqual(b'd', received)
        self.assertEqual(len(l), 5)
        expected = [b'b', b'c', b'e', b'f', b'g']
        received = [j for j in l]
        self.assertEqual(received, expected)

    def test_pop_index_error(self):
        l = List(self, 8, 0)
        with self.assertRaises(IndexError):
            l.pop()

    def test_pop_byte(self):
        l = List(self, 4, 0)
        values = [b'aaaa', b'bbbb', b'cccc', b'dddd', b'eeee', b'ffff', b'gggg', b'hhhhh']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        received = l.pop()
        self.assertEqual(b'hhhh', received)
        self.assertEqual(len(l), 7)
        received = [j for j in l]
        self.assertEqual(received, values[:-1])
        received = l.pop(0)
        self.assertEqual(b'aaaa', received)
        self.assertEqual(len(l), 6)
        received = l.pop(2)
        self.assertEqual(b'dddd', received)
        self.assertEqual(len(l), 5)
        expected = [b'bbbb', b'cccc', b'eeee', b'ffff', b'gggg']
        received = [j for j in l]
        self.assertEqual(received, expected)

    def test_delitem(self):
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        del l[0]
        self.assertEqual(len(l), 7)
        self.assertEqual(list(l), values[1:])
        del l[-1]
        self.assertEqual(len(l), 6)
        self.assertEqual(list(l), values[1:-1])
        del l[2]
        self.assertEqual(len(l), 5)
        self.assertEqual(list(l), [b'b', b'c', b'e', b'f', b'g'])

    def test_delete_slice(self):
        l = List(self, 1, 0)
        values = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
        for i in values:
            l.append(i)
        self.assertEqual(len(l), 8)
        del l[0:8:2]
        self.assertEqual(len(l), 4)
        self.assertEqual(list(l), values[1:8:2])
        del l[0:1:1]
        self.assertEqual(len(l), 3)
        self.assertEqual(list(l), [b'd', b'f', b'h'])
        del l[2:3:1]
        self.assertEqual(len(l), 2)
        self.assertEqual(list(l), [b'd', b'f'])
        del l[0:2:1]
        self.assertEqual(len(l), 0)
        self.assertEqual(list(l), [])

    def check_sizing(self, item_size, nmax):
        l = List(self, item_size, 0)

        def make_item(v):
            tmp = '{:0{}}'.format(nmax - v - 1, item_size).encode('latin-1')
            return tmp[:item_size]
        for i in range(nmax):
            l.append(make_item(i))
        self.assertEqual(len(l), nmax)
        for i in range(nmax):
            self.assertEqual(l[i], make_item(i))

    def test_sizing(self):
        for i in range(1, 16):
            self.check_sizing(item_size=i, nmax=2 ** i)

    def test_mutability(self):
        l = List(self, 8, 1)
        one = struct.pack('q', 1)
        l.append(one)
        self.assertTrue(l.is_mutable)
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)
        l.set_immutable()
        self.assertFalse(l.is_mutable)
        with self.assertRaises(ValueError) as raises:
            l.append(one)
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            l[0] = one
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            l.pop()
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            del l[0]
        self.assertIn('list is immutable', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            del l[0:1:1]
        self.assertIn('list is immutable', str(raises.exception))
        l.set_mutable()
        self.assertTrue(l.is_mutable)
        self.assertEqual(len(l), 1)
        r = struct.unpack('q', l[0])[0]
        self.assertEqual(r, 1)