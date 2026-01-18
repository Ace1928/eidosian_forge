from .. import tests, utextwrap
class TestUTextWrapAmbiWidth(tests.TestCase):
    _cyrill_char = 'А'

    def test_ambiwidth1(self):
        w = utextwrap.UTextWrapper(4, ambiguous_width=1)
        s = self._cyrill_char * 8
        self.assertEqual([self._cyrill_char * 4] * 2, w.wrap(s))

    def test_ambiwidth2(self):
        w = utextwrap.UTextWrapper(4, ambiguous_width=2)
        s = self._cyrill_char * 8
        self.assertEqual([self._cyrill_char * 2] * 4, w.wrap(s))