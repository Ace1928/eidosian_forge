import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
class SerialNumber8BitTests(unittest.TestCase):
    """
    Tests for correct answers to example calculations in RFC1982 5.2.

    Consider the case where SERIAL_BITS == 8.  In this space the integers that
    make up the serial number space are 0, 1, 2, ... 254, 255.  255 ==
    2^SERIAL_BITS - 1.

    https://tools.ietf.org/html/rfc1982#section-5.2
    """

    def test_maxadd(self):
        """
        In this space, the largest integer that it is meaningful to add to a
        sequence number is 2^(SERIAL_BITS - 1) - 1, or 127.
        """
        self.assertEqual(SerialNumber(0, serialBits=8)._maxAdd, 127)

    def test_add(self):
        """
        Addition is as expected in this space, for example: 255+1 == 0,
        100+100 == 200, and 200+100 == 44.
        """
        self.assertEqual(serialNumber8(255) + serialNumber8(1), serialNumber8(0))
        self.assertEqual(serialNumber8(100) + serialNumber8(100), serialNumber8(200))
        self.assertEqual(serialNumber8(200) + serialNumber8(100), serialNumber8(44))

    def test_gt(self):
        """
        Comparison is more interesting, 1 > 0, 44 > 0, 100 > 0, 100 > 44,
        200 > 100, 255 > 200, 0 > 255, 100 > 255, 0 > 200, and 44 > 200.
        """
        self.assertTrue(serialNumber8(1) > serialNumber8(0))
        self.assertTrue(serialNumber8(44) > serialNumber8(0))
        self.assertTrue(serialNumber8(100) > serialNumber8(0))
        self.assertTrue(serialNumber8(100) > serialNumber8(44))
        self.assertTrue(serialNumber8(200) > serialNumber8(100))
        self.assertTrue(serialNumber8(255) > serialNumber8(200))
        self.assertTrue(serialNumber8(100) > serialNumber8(255))
        self.assertTrue(serialNumber8(0) > serialNumber8(200))
        self.assertTrue(serialNumber8(44) > serialNumber8(200))

    def test_surprisingAddition(self):
        """
        Note that 100+100 > 100, but that (100+100)+100 < 100.  Incrementing a
        serial number can cause it to become "smaller".  Of course, incrementing
        by a smaller number will allow many more increments to be made before
        this occurs.  However this is always something to be aware of, it can
        cause surprising errors, or be useful as it is the only defined way to
        actually cause a serial number to decrease.
        """
        self.assertTrue(serialNumber8(100) + serialNumber8(100) > serialNumber8(100))
        self.assertTrue(serialNumber8(100) + serialNumber8(100) + serialNumber8(100) < serialNumber8(100))

    def test_undefined(self):
        """
        The pairs of values 0 and 128, 1 and 129, 2 and 130, etc, to 127 and 255
        are not equal, but in each pair, neither number is defined as being
        greater than, or less than, the other.
        """
        assertUndefinedComparison(self, serialNumber8(0), serialNumber8(128))
        assertUndefinedComparison(self, serialNumber8(1), serialNumber8(129))
        assertUndefinedComparison(self, serialNumber8(2), serialNumber8(130))
        assertUndefinedComparison(self, serialNumber8(127), serialNumber8(255))