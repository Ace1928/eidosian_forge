from io import BytesIO
from ...tests import TestCaseWithTransport
from ..unpeel_map import UnpeelMap
class TestUnpeelMap(TestCaseWithTransport):

    def test_new(self):
        m = UnpeelMap()
        self.assertIs(None, m.peel_tag('ab' * 20))

    def test_load(self):
        f = BytesIO(b'unpeel map version 1\n0123456789012345678901234567890123456789: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n')
        m = UnpeelMap()
        m.load(f)
        self.assertEqual(b'0123456789012345678901234567890123456789', m.peel_tag(b'aa' * 20))

    def test_update(self):
        m = UnpeelMap()
        m.update({b'0123456789012345678901234567890123456789': {b'aa' * 20}})
        self.assertEqual(b'0123456789012345678901234567890123456789', m.peel_tag(b'aa' * 20))