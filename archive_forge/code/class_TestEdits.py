import unittest
from bpython.curtsiesfrontend.manual_readline import (
class TestEdits(unittest.TestCase):

    def setUp(self):
        self.edits = UnconfiguredEdits()

    def test_seq(self):

        def f(cursor_offset, line):
            return ('hi', 2)
        self.edits.add('a', f)
        self.assertIn('a', self.edits)
        self.assertEqual(self.edits['a'], f)
        self.assertEqual(self.edits.call('a', cursor_offset=3, line='hello'), ('hi', 2))
        with self.assertRaises(KeyError):
            self.edits['b']
        with self.assertRaises(KeyError):
            self.edits.call('b')

    def test_functions_with_bad_signatures(self):

        def f(something):
            return (1, 2)
        with self.assertRaises(TypeError):
            self.edits.add('a', f)

        def g(cursor_offset, line, something, something_else):
            return (1, 2)
        with self.assertRaises(TypeError):
            self.edits.add('a', g)

    def test_functions_with_bad_return_values(self):

        def f(cursor_offset, line):
            return ('hi',)
        with self.assertRaises(ValueError):
            self.edits.add('a', f)

        def g(cursor_offset, line):
            return ('hi', 1, 2, 3)
        with self.assertRaises(ValueError):
            self.edits.add('b', g)

    def test_config(self):

        def f(cursor_offset, line):
            return ('hi', 2)

        def g(cursor_offset, line):
            return ('hey', 3)
        self.edits.add_config_attr('att', f)
        self.assertNotIn('att', self.edits)

        class config:
            att = 'c'
        key_dispatch = {'c': 'c'}
        configured_edits = self.edits.mapping_with_config(config, key_dispatch)
        self.assertTrue(configured_edits.__contains__, 'c')
        self.assertNotIn('c', self.edits)
        with self.assertRaises(NotImplementedError):
            configured_edits.add_config_attr('att2', g)
        with self.assertRaises(NotImplementedError):
            configured_edits.add('d', g)
        self.assertEqual(configured_edits.call('c', cursor_offset=5, line='asfd'), ('hi', 2))