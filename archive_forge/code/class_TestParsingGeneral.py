import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestParsingGeneral(unittest.TestCase):

    def test_parse_error(self):
        invalid_expressions = ['a:', '**', '.', '']
        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse(expression)

    def test_deep_nesting(self):
        actual = parse('[[a:b].c]:d')
        expected = trait('a', notify=False).trait('b').trait('c', notify=False).trait('d')
        self.assertEqual(actual, expected)
        actual = parse('[a:[b.[c:d]]]')
        expected = trait('a', notify=False).then(trait('b').then(trait('c', notify=False).then(trait('d'))))
        self.assertEqual(actual, expected)