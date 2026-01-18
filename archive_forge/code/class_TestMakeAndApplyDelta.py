import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
class TestMakeAndApplyDelta(tests.TestCase):
    scenarios = module_scenarios()
    _gc_module = None

    def setUp(self):
        super().setUp()
        self.make_delta = self._gc_module.make_delta
        self.apply_delta = self._gc_module.apply_delta
        self.apply_delta_to_source = self._gc_module.apply_delta_to_source

    def test_make_delta_is_typesafe(self):
        self.make_delta(b'a string', b'another string')

        def _check_make_delta(string1, string2):
            self.assertRaises(TypeError, self.make_delta, string1, string2)
        _check_make_delta(b'a string', object())
        _check_make_delta(b'a string', 'not a string')
        _check_make_delta(object(), b'a string')
        _check_make_delta('not a string', b'a string')

    def test_make_noop_delta(self):
        ident_delta = self.make_delta(_text1, _text1)
        self.assertEqual(b'M\x90M', ident_delta)
        ident_delta = self.make_delta(_text2, _text2)
        self.assertEqual(b'N\x90N', ident_delta)
        ident_delta = self.make_delta(_text3, _text3)
        self.assertEqual(b'\x87\x01\x90\x87', ident_delta)

    def assertDeltaIn(self, delta1, delta2, delta):
        """Make sure that the delta bytes match one of the expectations."""
        if delta not in (delta1, delta2):
            self.fail(b'Delta bytes:\n       %r\nnot in %r\n    or %r' % (delta, delta1, delta2))

    def test_make_delta(self):
        delta = self.make_delta(_text1, _text2)
        self.assertDeltaIn(b'N\x90/\x1fdiffer from\nagainst other text\n', b'N\x90\x1d\x1ewhich is meant to differ from\n\x91:\x13', delta)
        delta = self.make_delta(_text2, _text1)
        self.assertDeltaIn(b'M\x90/\x1ebe matched\nagainst other text\n', b'M\x90\x1d\x1dwhich is meant to be matched\n\x91;\x13', delta)
        delta = self.make_delta(_text3, _text1)
        self.assertEqual(b'M\x90M', delta)
        delta = self.make_delta(_text3, _text2)
        self.assertDeltaIn(b'N\x90/\x1fdiffer from\nagainst other text\n', b'N\x90\x1d\x1ewhich is meant to differ from\n\x91:\x13', delta)

    def test_make_delta_with_large_copies(self):
        big_text = _text3 * 1220
        delta = self.make_delta(big_text, big_text)
        self.assertDeltaIn(b'\xdc\x86\n\x80\x84\x01\xb4\x02\\\x83', None, delta)

    def test_apply_delta_is_typesafe(self):
        self.apply_delta(_text1, b'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, object(), b'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, _text1.decode('latin1'), b'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, _text1, 'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, _text1, object())

    def test_apply_delta(self):
        target = self.apply_delta(_text1, b'N\x90/\x1fdiffer from\nagainst other text\n')
        self.assertEqual(_text2, target)
        target = self.apply_delta(_text2, b'M\x90/\x1ebe matched\nagainst other text\n')
        self.assertEqual(_text1, target)

    def test_apply_delta_to_source_is_safe(self):
        self.assertRaises(TypeError, self.apply_delta_to_source, object(), 0, 1)
        self.assertRaises(TypeError, self.apply_delta_to_source, 'unicode str', 0, 1)
        self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 1, 4)
        self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 5, 3)
        self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 3, 2)

    def test_apply_delta_to_source(self):
        source_and_delta = _text1 + b'N\x90/\x1fdiffer from\nagainst other text\n'
        self.assertEqual(_text2, self.apply_delta_to_source(source_and_delta, len(_text1), len(source_and_delta)))