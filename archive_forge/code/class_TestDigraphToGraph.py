import testtools
import testresources
from testresources import split_by_resources, _resource_graph
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestDigraphToGraph(testtools.TestCase):

    def test_wikipedia_example(self):
        """Converting a digraph mirrors it in the XZ axis (matrix view).

        See http://en.wikipedia.org/wiki/Travelling_salesman_problem         #Solving_by_conversion_to_Symmetric_TSP
        """
        A = 'A'
        Ap = "A'"
        B = 'B'
        Bp = "B'"
        C = 'C'
        Cp = "C'"
        digraph = {A: {B: 1, C: 2}, B: {A: 6, C: 3}, C: {A: 5, B: 4}}
        expected = {A: {Ap: 0, Bp: 6, Cp: 5}, B: {Ap: 1, Bp: 0, Cp: 4}, C: {Ap: 2, Bp: 3, Cp: 0}, Ap: {A: 0, B: 1, C: 2}, Bp: {A: 6, B: 0, C: 3}, Cp: {A: 5, B: 4, C: 0}}
        self.assertEqual(expected, testresources._digraph_to_graph(digraph, {A: Ap, B: Bp, C: Cp}))