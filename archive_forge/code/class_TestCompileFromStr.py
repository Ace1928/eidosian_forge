import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestCompileFromStr(unittest.TestCase):

    def test_compile_simple(self):
        actual = compile_str('name')
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False))]
        self.assertEqual(actual, expected)

    def test_compile_serial(self):
        actual = compile_str('name1.name2')
        expected = [create_graph(NamedTraitObserver(name='name1', notify=True, optional=False), NamedTraitObserver(name='name2', notify=True, optional=False))]
        self.assertEqual(actual, expected)

    def test_compile_parallel(self):
        actual = compile_str('name1,name2')
        expected = [create_graph(NamedTraitObserver(name='name1', notify=True, optional=False)), create_graph(NamedTraitObserver(name='name2', notify=True, optional=False))]
        self.assertEqual(actual, expected)