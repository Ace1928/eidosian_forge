import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestOptimisingTestSuite(testtools.TestCase):

    def makeTestCase(self, test_running_hook=None):
        """Make a normal TestCase."""

        class TestCaseForTesting(unittest.TestCase):

            def runTest(self):
                if test_running_hook:
                    test_running_hook(self)
        return TestCaseForTesting('runTest')

    def makeResourcedTestCase(self, resource_manager, test_running_hook):
        """Make a ResourcedTestCase."""

        class ResourcedTestCaseForTesting(testresources.ResourcedTestCase):

            def runTest(self):
                test_running_hook(self)
        test_case = ResourcedTestCaseForTesting('runTest')
        test_case.resources = [('_default', resource_manager)]
        return test_case

    def setUp(self):
        super(TestOptimisingTestSuite, self).setUp()
        self.optimising_suite = testresources.OptimisingTestSuite()

    def testAddTest(self):
        case = self.makeTestCase()
        self.optimising_suite.addTest(case)
        self.assertEqual([case], self.optimising_suite._tests)

    def testAddTestSuite(self):
        case = self.makeTestCase()
        suite = unittest.TestSuite([case])
        self.optimising_suite.addTest(suite)
        self.assertEqual([case], self.optimising_suite._tests)

    @testtools.skipIf(unittest2 is None, 'Unittest2 needed')
    def testAddUnittest2TestSuite(self):
        case = self.makeTestCase()
        suite = unittest2.TestSuite([case])
        self.optimising_suite.addTest(suite)
        self.assertEqual([case], self.optimising_suite._tests)

    def testAddTestOptimisingTestSuite(self):
        case = self.makeTestCase()
        suite1 = testresources.OptimisingTestSuite([case])
        suite2 = testresources.OptimisingTestSuite([case])
        self.optimising_suite.addTest(suite1)
        self.optimising_suite.addTest(suite2)
        self.assertEqual([case, case], self.optimising_suite._tests)

    def testAddFlattensStandardSuiteStructure(self):
        case1 = self.makeTestCase()
        case2 = self.makeTestCase()
        case3 = self.makeTestCase()
        suite = unittest.TestSuite([unittest.TestSuite([case1, unittest.TestSuite([case2])]), case3])
        self.optimising_suite.addTest(suite)
        self.assertEqual([case1, case2, case3], self.optimising_suite._tests)

    def testAddDistributesNonStandardSuiteStructure(self):
        case1 = self.makeTestCase()
        case2 = self.makeTestCase()
        inner_suite = unittest.TestSuite([case2])
        suite = CustomSuite([case1, inner_suite])
        self.optimising_suite.addTest(suite)
        self.assertEqual([CustomSuite([case1]), CustomSuite([inner_suite])], self.optimising_suite._tests)

    def testAddPullsNonStandardSuitesUp(self):
        case1 = self.makeTestCase()
        case2 = self.makeTestCase()
        inner_suite = CustomSuite([case1, case2])
        self.optimising_suite.addTest(unittest.TestSuite([unittest.TestSuite([inner_suite])]))
        self.assertEqual([CustomSuite([case1]), CustomSuite([case2])], self.optimising_suite._tests)

    def testSingleCaseResourceAcquisition(self):
        sample_resource = MakeCounter()

        def getResourceCount(test):
            self.assertEqual(sample_resource._uses, 2)
        case = self.makeResourcedTestCase(sample_resource, getResourceCount)
        self.optimising_suite.addTest(case)
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.wasSuccessful(), True)
        self.assertEqual(sample_resource._uses, 0)

    def testResourceReuse(self):
        make_counter = MakeCounter()

        def getResourceCount(test):
            self.assertEqual(make_counter._uses, 2)
        case = self.makeResourcedTestCase(make_counter, getResourceCount)
        case2 = self.makeResourcedTestCase(make_counter, getResourceCount)
        self.optimising_suite.addTest(case)
        self.optimising_suite.addTest(case2)
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(result.wasSuccessful(), True)
        self.assertEqual(make_counter._uses, 0)
        self.assertEqual(make_counter.makes, 1)
        self.assertEqual(make_counter.cleans, 1)

    def testResultPassedToResources(self):
        resource_manager = MakeCounter()
        test_case = self.makeTestCase(lambda x: None)
        test_case.resources = [('_default', resource_manager)]
        self.optimising_suite.addTest(test_case)
        result = ResultWithResourceExtensions()
        self.optimising_suite.run(result)
        self.assertEqual(4, len(result._calls))

    def testOptimisedRunNonResourcedTestCase(self):
        case = self.makeTestCase()
        self.optimising_suite.addTest(case)
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.wasSuccessful(), True)

    def testSortTestsCalled(self):

        class MockOptimisingTestSuite(testresources.OptimisingTestSuite):

            def sortTests(self):
                self.sorted = True
        suite = MockOptimisingTestSuite()
        suite.sorted = False
        suite.run(None)
        self.assertEqual(suite.sorted, True)

    def testResourcesDroppedForNonResourcedTestCase(self):
        sample_resource = MakeCounter()

        def resourced_case_hook(test):
            self.assertTrue(sample_resource._uses > 0)
        self.optimising_suite.addTest(self.makeResourcedTestCase(sample_resource, resourced_case_hook))

        def normal_case_hook(test):
            self.assertEqual(sample_resource._uses, 0)
        self.optimising_suite.addTest(self.makeTestCase(normal_case_hook))
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual([], result.failures)
        self.assertEqual([], result.errors)
        self.assertEqual(result.wasSuccessful(), True)

    def testDirtiedResourceNotRecreated(self):
        make_counter = MakeCounter()

        def dirtyResource(test):
            make_counter.dirtied(test._default)
        case = self.makeResourcedTestCase(make_counter, dirtyResource)
        self.optimising_suite.addTest(case)
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.wasSuccessful(), True)
        self.assertEqual(make_counter.makes, 1)

    def testDirtiedResourceCleanedUp(self):
        make_counter = MakeCounter()

        def testOne(test):
            make_counter.calls.append('test one')
            make_counter.dirtied(test._default)

        def testTwo(test):
            make_counter.calls.append('test two')
        case1 = self.makeResourcedTestCase(make_counter, testOne)
        case2 = self.makeResourcedTestCase(make_counter, testTwo)
        self.optimising_suite.addTest(case1)
        self.optimising_suite.addTest(case2)
        result = unittest.TestResult()
        self.optimising_suite.run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(result.wasSuccessful(), True)
        self.assertEqual(make_counter.calls, [('make', 'boo 1'), 'test one', ('clean', 'boo 1'), ('make', 'boo 2'), 'test two', ('clean', 'boo 2')])