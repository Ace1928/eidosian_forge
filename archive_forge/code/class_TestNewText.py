from unittest import TestCase
import patiencediff
from .. import multiparent, tests
class TestNewText(TestCase):

    def test_eq(self):
        self.assertEqual(multiparent.NewText([]), multiparent.NewText([]))
        self.assertFalse(multiparent.NewText([b'a']) == multiparent.NewText([b'b']))
        self.assertFalse(multiparent.NewText([b'a']) == Mock(lines=[b'a']))

    def test_to_patch(self):
        self.assertEqual([b'i 0\n', b'\n'], list(multiparent.NewText([]).to_patch()))
        self.assertEqual([b'i 1\n', b'a', b'\n'], list(multiparent.NewText([b'a']).to_patch()))
        self.assertEqual([b'i 1\n', b'a\n', b'\n'], list(multiparent.NewText([b'a\n']).to_patch()))