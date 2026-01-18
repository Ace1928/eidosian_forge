import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestCostGraph(testtools.TestCase):
    """Tests for calculating the cost graph of resourced test cases."""

    def makeResource(self, setUpCost=1, tearDownCost=1):
        resource = testresources.TestResource()
        resource.setUpCost = setUpCost
        resource.tearDownCost = tearDownCost
        return resource

    def testEmptyGraph(self):
        suite = testresources.OptimisingTestSuite()
        graph = suite._getGraph([])
        self.assertEqual({}, graph)

    def testSingletonGraph(self):
        resource = self.makeResource()
        suite = testresources.OptimisingTestSuite()
        graph = suite._getGraph([frozenset()])
        self.assertEqual({frozenset(): {}}, graph)

    def testTwoCasesInGraph(self):
        res1 = self.makeResource()
        res2 = self.makeResource()
        set1 = frozenset([res1, res2])
        set2 = frozenset([res2])
        no_resources = frozenset()
        suite = testresources.OptimisingTestSuite()
        graph = suite._getGraph([no_resources, set1, set2])
        self.assertEqual({no_resources: {set1: 2, set2: 1}, set1: {no_resources: 2, set2: 1}, set2: {no_resources: 1, set1: 1}}, graph)