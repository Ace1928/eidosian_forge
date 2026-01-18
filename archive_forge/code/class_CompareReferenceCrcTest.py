import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class CompareReferenceCrcTest(unittest.TestCase):
    test_messages = [b'', b'T', b'123456789', b'CatMouse987654321']
    test_poly_crcs = [[(g8, 0, 0), crc8p], [(g16, 0, 0), crc16p], [(g24, 0, 0), crc24p], [(g32, 0, 0), crc32p], [(g64a, 0, 0), crc64ap], [(g64b, 0, 0), crc64bp]]

    @staticmethod
    def reference_crc32(d, crc=0):
        """This function modifies the return value of binascii.crc32
        to be an unsigned 32-bit value. I.e. in the range 0 to 2**32-1."""
        if crc > 2147483647:
            x = int(crc & 2147483647)
            crc = x | -2147483648
        x = binascii.crc32(d, crc)
        return int(x) & 4294967295

    def test_compare_crc32(self):
        """The binascii module has a 32-bit CRC function that is used in a wide range
        of applications including the checksum used in the ZIP file format.
        This test compares the CRC-32 implementation of this crcmod module to
        that of binascii.crc32."""
        crc32 = mkCrcFun(g32, 0, 1, 4294967295)
        for msg in self.test_messages:
            self.assertEqual(crc32(msg), self.reference_crc32(msg))

    def test_compare_poly(self):
        """Compare various CRCs of this crcmod module to a pure
        polynomial-based implementation."""
        for crcfun_params, crc_poly_fun in self.test_poly_crcs:
            crcfun = mkCrcFun(*crcfun_params)
            for msg in self.test_messages:
                self.assertEqual(crcfun(msg), crc_poly_fun(msg))