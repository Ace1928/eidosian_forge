import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
class TestResourcedTestCase(testtools.TestCase):

    def setUp(self):
        super(TestResourcedTestCase, self).setUp()

        class Example(testresources.ResourcedTestCase):

            def test_example(self):
                pass
        self.resourced_case = Example('test_example')
        self.resource = self.getUniqueString()
        self.resource_manager = MockResource(self.resource)

    def testSetUpUsesSuper(self):

        class OtherBaseCase(unittest.TestCase):
            setUpCalled = False

            def setUp(self):
                self.setUpCalled = True
                super(OtherBaseCase, self).setUp()

        class OurCase(testresources.ResourcedTestCase, OtherBaseCase):

            def runTest(self):
                pass
        ourCase = OurCase()
        ourCase.setUp()
        self.assertTrue(ourCase.setUpCalled)

    def testTearDownUsesSuper(self):

        class OtherBaseCase(unittest.TestCase):
            tearDownCalled = False

            def tearDown(self):
                self.tearDownCalled = True
                super(OtherBaseCase, self).setUp()

        class OurCase(testresources.ResourcedTestCase, OtherBaseCase):

            def runTest(self):
                pass
        ourCase = OurCase()
        ourCase.setUp()
        ourCase.tearDown()
        self.assertTrue(ourCase.tearDownCalled)

    def testDefaults(self):
        self.assertEqual(self.resourced_case.resources, [])

    def testResultPassedToResources(self):
        result = ResultWithResourceExtensions()
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resourced_case.run(result)
        self.assertEqual(4, len(result._calls))

    def testSetUpResourcesSingle(self):
        self.resourced_case.resources = [('foo', self.resource_manager)]
        testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
        self.assertEqual(self.resource, self.resourced_case.foo)

    def testSetUpResourcesMultiple(self):
        self.resourced_case.resources = [('foo', self.resource_manager), ('bar', MockResource('bar_resource'))]
        testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
        self.assertEqual(self.resource, self.resourced_case.foo)
        self.assertEqual('bar_resource', self.resourced_case.bar)

    def testSetUpResourcesSetsUpDependences(self):
        resource = MockResourceInstance()
        self.resource_manager = MockResource(resource)
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resource_manager.resources.append(('bar', MockResource('bar_resource')))
        testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
        self.assertEqual(resource, self.resourced_case.foo)
        self.assertEqual('bar_resource', self.resourced_case.foo.bar)

    def testSetUpUsesResource(self):
        self.resourced_case.resources = [('foo', self.resource_manager)]
        testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
        self.assertEqual(self.resource_manager._uses, 1)

    def testTearDownResourcesDeletesResourceAttributes(self):
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resourced_case.setUpResources()
        self.resourced_case.tearDownResources()
        self.failIf(hasattr(self.resourced_case, 'foo'))

    def testTearDownResourcesStopsUsingResource(self):
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resourced_case.setUpResources()
        self.resourced_case.tearDownResources()
        self.assertEqual(self.resource_manager._uses, 0)

    def testTearDownResourcesStopsUsingDependencies(self):
        resource = MockResourceInstance()
        dep1 = MockResource('bar_resource')
        self.resource_manager = MockResource(resource)
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resource_manager.resources.append(('bar', dep1))
        self.resourced_case.setUpResources()
        self.resourced_case.tearDownResources()
        self.assertEqual(dep1._uses, 0)

    def testSingleWithSetup(self):
        self.resourced_case.resources = [('foo', self.resource_manager)]
        self.resourced_case.setUp()
        self.assertEqual(self.resourced_case.foo, self.resource)
        self.assertEqual(self.resource_manager._uses, 1)
        self.resourced_case.tearDown()
        self.failIf(hasattr(self.resourced_case, 'foo'))
        self.assertEqual(self.resource_manager._uses, 0)