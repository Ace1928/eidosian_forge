import sys
from tests.base import BaseTestCase
from pyasn1.compat import binary
class BinaryTestCase(BaseTestCase):

    def test_bin_zero(self):
        assert '0b0' == binary.bin(0)

    def test_bin_noarg(self):
        try:
            binary.bin()
        except TypeError:
            pass
        except:
            assert 0, 'bin() tolerates no arguments'

    def test_bin_allones(self):
        assert '0b1111111111111111111111111111111111111111111111111111111111111111' == binary.bin(18446744073709551615)

    def test_bin_allzeros(self):
        assert '0b0' == binary.bin(0)

    def test_bin_pos(self):
        assert '0b1000000010000000100000001' == binary.bin(16843009)

    def test_bin_neg(self):
        assert '-0b1000000010000000100000001' == binary.bin(-16843009)