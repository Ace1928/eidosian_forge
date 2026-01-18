import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestGraphStuff(testtools.TestCase):

    def setUp(self):
        super(TestGraphStuff, self).setUp()

        class MockTest(unittest.TestCase):

            def __repr__(self):
                """The representation is the tests name.

                This makes it easier to debug sorting failures.
                """
                return self.id().split('.')[-1]

            def test_one(self):
                pass

            def test_two(self):
                pass

            def test_three(self):
                pass

            def test_four(self):
                pass
        self.case1 = MockTest('test_one')
        self.case2 = MockTest('test_two')
        self.case3 = MockTest('test_three')
        self.case4 = MockTest('test_four')
        self.cases = []
        self.cases.append(self.case1)
        self.cases.append(self.case2)
        self.cases.append(self.case3)
        self.cases.append(self.case4)

    def sortTests(self, tests):
        suite = testresources.OptimisingTestSuite()
        suite.addTests(tests)
        suite.sortTests()
        return suite._tests

    def _permute_four(self, cases):
        case1, case2, case3, case4 = cases
        permutations = []
        permutations.append([case1, case2, case3, case4])
        permutations.append([case1, case2, case4, case3])
        permutations.append([case1, case3, case2, case4])
        permutations.append([case1, case3, case4, case2])
        permutations.append([case1, case4, case2, case3])
        permutations.append([case1, case4, case3, case2])
        permutations.append([case2, case1, case3, case4])
        permutations.append([case2, case1, case4, case3])
        permutations.append([case2, case3, case1, case4])
        permutations.append([case2, case3, case4, case1])
        permutations.append([case2, case4, case1, case3])
        permutations.append([case2, case4, case3, case1])
        permutations.append([case3, case2, case1, case4])
        permutations.append([case3, case2, case4, case1])
        permutations.append([case3, case1, case2, case4])
        permutations.append([case3, case1, case4, case2])
        permutations.append([case3, case4, case2, case1])
        permutations.append([case3, case4, case1, case2])
        permutations.append([case4, case2, case3, case1])
        permutations.append([case4, case2, case1, case3])
        permutations.append([case4, case3, case2, case1])
        permutations.append([case4, case3, case1, case2])
        permutations.append([case4, case1, case2, case3])
        permutations.append([case4, case1, case3, case2])
        return permutations

    def testBasicSortTests(self):
        resource_one = testresources.TestResource()
        resource_two = testresources.TestResource()
        resource_two.setUpCost = 5
        resource_two.tearDownCost = 5
        resource_three = testresources.TestResource()
        self.case1.resources = [('_one', resource_one), ('_two', resource_two)]
        self.case2.resources = [('_two', resource_two), ('_three', resource_three)]
        self.case3.resources = [('_three', resource_three)]
        for permutation in self._permute_four(self.cases):
            self.assertIn(self.sortTests(permutation), [[self.case1, self.case2, self.case3, self.case4], [self.case3, self.case2, self.case1, self.case4]], 'failed with permutation %s' % (permutation,))

    def testGlobalMinimum(self):
        resource_one = testresources.TestResource()
        resource_one.setUpCost = 20
        resource_two = testresources.TestResource()
        resource_two.tearDownCost = 50
        resource_three = testresources.TestResource()
        resource_three.setUpCost = 72
        acceptable_orders = [[self.case1, self.case2, self.case3, self.case4], [self.case1, self.case3, self.case2, self.case4], [self.case2, self.case3, self.case1, self.case4], [self.case3, self.case2, self.case1, self.case4]]
        self.case1.resources = [('_one', resource_one)]
        self.case2.resources = [('_two', resource_two)]
        self.case3.resources = [('_two', resource_two), ('_three', resource_three)]
        for permutation in self._permute_four(self.cases):
            self.assertIn(self.sortTests(permutation), acceptable_orders)

    def testSortIsStableWithinGroups(self):
        """Tests with the same resources maintain their relative order."""
        resource_one = testresources.TestResource()
        resource_two = testresources.TestResource()
        self.case1.resources = [('_one', resource_one)]
        self.case2.resources = [('_one', resource_one)]
        self.case3.resources = [('_one', resource_one), ('_two', resource_two)]
        self.case4.resources = [('_one', resource_one), ('_two', resource_two)]
        for permutation in self._permute_four(self.cases):
            sorted = self.sortTests(permutation)
            self.assertEqual(permutation.index(self.case1) < permutation.index(self.case2), sorted.index(self.case1) < sorted.index(self.case2))
            self.assertEqual(permutation.index(self.case3) < permutation.index(self.case4), sorted.index(self.case3) < sorted.index(self.case4))

    def testSortingTwelveIndependentIsFast(self):
        managers = []
        for pos in range(12):
            managers.append(testresources.TestResourceManager())
        cases = [self.case1, self.case2, self.case3, self.case4]
        for pos in range(5, 13):
            cases.append(testtools.clone_test_with_new_id(cases[0], 'case%d' % pos))
        for case, manager in zip(cases, managers):
            case.resources = [('_resource', manager)]
        result = self.sortTests(cases)
        self.assertEqual(12, len(result))

    def testSortingTwelveOverlappingIsFast(self):
        managers = []
        for pos in range(12):
            managers.append(testresources.TestResourceManager())
        cases = [self.case1, self.case2, self.case3, self.case4]
        for pos in range(5, 13):
            cases.append(testtools.clone_test_with_new_id(cases[0], 'case%d' % pos))
        tempdir = testresources.TestResourceManager()
        for case, manager in zip(cases, managers):
            case.resources = [('_resource', manager), ('tempdir', tempdir)]
        result = self.sortTests(cases)
        self.assertEqual(12, len(result))

    def testSortConsidersDependencies(self):
        """Tests with different dependencies are sorted together."""
        resource_one = testresources.TestResource()
        resource_two = testresources.TestResource()
        resource_one_common = testresources.TestResource()
        resource_one_common.setUpCost = 2
        resource_one_common.tearDownCost = 2
        resource_two_common = testresources.TestResource()
        resource_two_common.setUpCost = 2
        resource_two_common.tearDownCost = 2
        dep = testresources.TestResource()
        dep.setUpCost = 20
        dep.tearDownCost = 20
        resource_one.resources.append(('dep1', dep))
        resource_two.resources.append(('dep2', dep))
        self.case1.resources = [('withdep', resource_one), ('common', resource_one_common)]
        self.case2.resources = [('withdep', resource_two), ('common', resource_two_common)]
        self.case3.resources = [('_one', resource_one_common), ('_two', resource_two_common)]
        self.case4.resources = []
        acceptable_orders = [[self.case1, self.case2, self.case3, self.case4], [self.case2, self.case1, self.case3, self.case4], [self.case3, self.case1, self.case2, self.case4], [self.case3, self.case2, self.case1, self.case4]]
        for permutation in self._permute_four(self.cases):
            self.assertIn(self.sortTests(permutation), acceptable_orders)