import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class TestClassCleanup(unittest.TestCase):

    def test_addClassCleanUp(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        test = TestableTest('testNothing')
        self.assertEqual(test._class_cleanups, [])
        class_cleanups = []

        def class_cleanup1(*args, **kwargs):
            class_cleanups.append((3, args, kwargs))

        def class_cleanup2(*args, **kwargs):
            class_cleanups.append((4, args, kwargs))
        TestableTest.addClassCleanup(class_cleanup1, 1, 2, 3, four='hello', five='goodbye')
        TestableTest.addClassCleanup(class_cleanup2)
        self.assertEqual(test._class_cleanups, [(class_cleanup1, (1, 2, 3), dict(four='hello', five='goodbye')), (class_cleanup2, (), {})])
        TestableTest.doClassCleanups()
        self.assertEqual(class_cleanups, [(4, (), {}), (3, (1, 2, 3), dict(four='hello', five='goodbye'))])

    def test_run_class_cleanUp(self):
        ordering = []
        blowUp = True

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering)
                if blowUp:
                    raise CustomError()

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpClass', 'cleanup_good'])
        ordering = []
        blowUp = False
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_good'])

    def test_run_class_cleanUp_without_tearDownClass(self):
        ordering = []
        blowUp = True

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering)
                if blowUp:
                    raise CustomError()

            def testNothing(self):
                ordering.append('test')

            @classmethod
            @property
            def tearDownClass(cls):
                raise AttributeError
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpClass', 'cleanup_good'])
        ordering = []
        blowUp = False
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpClass', 'test', 'cleanup_good'])

    def test_debug_executes_classCleanUp(self):
        ordering = []
        blowUp = False

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering, blowUp=blowUp)

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        suite.debug()
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_good'])
        ordering = []
        blowUp = True
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_exc'])

    def test_debug_executes_classCleanUp_when_teardown_exception(self):
        ordering = []
        blowUp = False

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering, blowUp=blowUp)

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
                raise CustomError('TearDownClassExc')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'TearDownClassExc')
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass'])
        self.assertTrue(TestableTest._class_cleanups)
        TestableTest._class_cleanups.clear()
        ordering = []
        blowUp = True
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'TearDownClassExc')
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass'])
        self.assertTrue(TestableTest._class_cleanups)
        TestableTest._class_cleanups.clear()

    def test_doClassCleanups_with_errors_addClassCleanUp(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass

        def cleanup1():
            raise CustomError('cleanup1')

        def cleanup2():
            raise CustomError('cleanup2')
        TestableTest.addClassCleanup(cleanup1)
        TestableTest.addClassCleanup(cleanup2)
        TestableTest.doClassCleanups()
        self.assertEqual(len(TestableTest.tearDown_exceptions), 2)
        e1, e2 = TestableTest.tearDown_exceptions
        self.assertIsInstance(e1[1], CustomError)
        self.assertEqual(str(e1[1]), 'cleanup2')
        self.assertIsInstance(e2[1], CustomError)
        self.assertEqual(str(e2[1]), 'cleanup1')

    def test_with_errors_addCleanUp(self):
        ordering = []

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering)

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup, ordering, blowUp=True)

            def testNothing(self):
                pass

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'setUp', 'cleanup_exc', 'tearDownClass', 'cleanup_good'])

    def test_run_with_errors_addClassCleanUp(self):
        ordering = []

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering, blowUp=True)

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup, ordering)

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'setUp', 'test', 'cleanup_good', 'tearDownClass', 'cleanup_exc'])

    def test_with_errors_in_addClassCleanup_and_setUps(self):
        ordering = []
        class_blow_up = False
        method_blow_up = False

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering, blowUp=True)
                if class_blow_up:
                    raise CustomError('ClassExc')

            def setUp(self):
                ordering.append('setUp')
                if method_blow_up:
                    raise CustomError('MethodExc')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'setUp', 'test', 'tearDownClass', 'cleanup_exc'])
        ordering = []
        class_blow_up = True
        method_blow_up = False
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ClassExc')
        self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'cleanup_exc'])
        ordering = []
        class_blow_up = False
        method_blow_up = True
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: MethodExc')
        self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpClass', 'setUp', 'tearDownClass', 'cleanup_exc'])

    def test_with_errors_in_tearDownClass(self):
        ordering = []

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering)

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
                raise CustomError('TearDownExc')
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: TearDownExc')
        self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass', 'cleanup_good'])

    def test_enterClassContext(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        cleanups = []
        TestableTest.addClassCleanup(cleanups.append, 'cleanup1')
        cm = TestCM(cleanups, 42)
        self.assertEqual(TestableTest.enterClassContext(cm), 42)
        TestableTest.addClassCleanup(cleanups.append, 'cleanup2')
        TestableTest.doClassCleanups()
        self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])

    def test_enterClassContext_arg_errors(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            TestableTest.enterClassContext(LacksEnterAndExit())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            TestableTest.enterClassContext(LacksEnter())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            TestableTest.enterClassContext(LacksExit())
        self.assertEqual(TestableTest._class_cleanups, [])

    def test_run_nested_test(self):
        ordering = []

        class InnerTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('inner setup')
                cls.addClassCleanup(ordering.append, 'inner cleanup')

            def test(self):
                ordering.append('inner test')

        class OuterTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('outer setup')
                cls.addClassCleanup(ordering.append, 'outer cleanup')

            def test(self):
                ordering.append('start outer test')
                runTests(InnerTest)
                ordering.append('end outer test')
        runTests(OuterTest)
        self.assertEqual(ordering, ['outer setup', 'start outer test', 'inner setup', 'inner test', 'inner cleanup', 'end outer test', 'outer cleanup'])