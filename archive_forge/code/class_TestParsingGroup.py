import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestParsingGroup(unittest.TestCase):

    def test_grouped_or(self):
        actual = parse('root.[left,right]')
        expected = trait('root').then(trait('left') | trait('right'))
        self.assertEqual(actual, expected)

    def test_grouped_or_extended(self):
        actual = parse('root.[left,right].value')
        expected = trait('root').then(trait('left') | trait('right')).trait('value')
        self.assertEqual(actual, expected)

    def test_multi_branch_then_or_apply_notify_flag_to_last_item(self):
        actual = parse('root.[a.b.c.d,value]:g')
        expected = trait('root').then(trait('a').trait('b').trait('c').trait('d', False) | trait('value', False)).trait('g')
        self.assertEqual(actual, expected)