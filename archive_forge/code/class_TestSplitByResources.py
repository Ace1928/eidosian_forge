import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestSplitByResources(testtools.TestCase):
    """Tests for split_by_resources."""

    def makeTestCase(self):
        return unittest.TestCase('run')

    def makeResourcedTestCase(self, has_resource=True):
        case = testresources.ResourcedTestCase('run')
        if has_resource:
            case.resources = [('resource', testresources.TestResource())]
        return case

    def testNoTests(self):
        self.assertEqual({frozenset(): []}, split_by_resources([]))

    def testJustNormalCases(self):
        normal_case = self.makeTestCase()
        resource_set_tests = split_by_resources([normal_case])
        self.assertEqual({frozenset(): [normal_case]}, resource_set_tests)

    def testJustResourcedCases(self):
        resourced_case = self.makeResourcedTestCase()
        resource = resourced_case.resources[0][1]
        resource_set_tests = split_by_resources([resourced_case])
        self.assertEqual({frozenset(): [], frozenset([resource]): [resourced_case]}, resource_set_tests)

    def testMultipleResources(self):
        resource1 = testresources.TestResource()
        resource2 = testresources.TestResource()
        resourced_case = self.makeResourcedTestCase(has_resource=False)
        resourced_case.resources = [('resource1', resource1), ('resource2', resource2)]
        resource_set_tests = split_by_resources([resourced_case])
        self.assertEqual({frozenset(): [], frozenset([resource1, resource2]): [resourced_case]}, resource_set_tests)

    def testDependentResources(self):
        resource1 = testresources.TestResource()
        resource2 = testresources.TestResource()
        resource1.resources = [('foo', resource2)]
        resourced_case = self.makeResourcedTestCase(has_resource=False)
        resourced_case.resources = [('resource1', resource1)]
        resource_set_tests = split_by_resources([resourced_case])
        self.assertEqual({frozenset(): [], frozenset([resource1, resource2]): [resourced_case]}, resource_set_tests)

    def testResourcedCaseWithNoResources(self):
        resourced_case = self.makeResourcedTestCase(has_resource=False)
        resource_set_tests = split_by_resources([resourced_case])
        self.assertEqual({frozenset(): [resourced_case]}, resource_set_tests)

    def testMixThemUp(self):
        normal_cases = [self.makeTestCase() for i in range(3)]
        normal_cases.extend([self.makeResourcedTestCase(has_resource=False) for i in range(3)])
        resourced_cases = [self.makeResourcedTestCase() for i in range(3)]
        all_cases = normal_cases + resourced_cases
        random.shuffle(all_cases)
        resource_set_tests = split_by_resources(all_cases)
        self.assertEqual(set(normal_cases), set(resource_set_tests[frozenset()]))
        for case in resourced_cases:
            resource = case.resources[0][1]
            self.assertEqual([case], resource_set_tests[frozenset([resource])])