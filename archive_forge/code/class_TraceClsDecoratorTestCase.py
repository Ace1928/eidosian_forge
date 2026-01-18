import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class TraceClsDecoratorTestCase(test.TestCase):

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_args(self, mock_start, mock_stop):
        fake_cls = FakeTraceClassWithInfo()
        self.assertEqual(30, fake_cls.method1(5, 15))
        expected_info = {'a': 10, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceClassWithInfo.method1', 'args': str((fake_cls, 5, 15)), 'kwargs': str({})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_kwargs(self, mock_start, mock_stop):
        fake_cls = FakeTraceClassWithInfo()
        self.assertEqual(50, fake_cls.method3(g=5, h=10))
        expected_info = {'a': 10, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceClassWithInfo.method3', 'args': str((fake_cls,)), 'kwargs': str({'g': 5, 'h': 10})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_without_private(self, mock_start, mock_stop):
        fake_cls = FakeTraceClassHideArgs()
        self.assertEqual(10, fake_cls._method(10))
        self.assertFalse(mock_start.called)
        self.assertFalse(mock_stop.called)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_without_args(self, mock_start, mock_stop):
        fake_cls = FakeTraceClassHideArgs()
        self.assertEqual(40, fake_cls.method1(5, 15, c=20))
        expected_info = {'b': 20, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceClassHideArgs.method1'}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('a', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_private_methods(self, mock_start, mock_stop):
        fake_cls = FakeTracePrivate()
        self.assertEqual(5, fake_cls._method(5))
        expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTracePrivate._method', 'args': str((fake_cls, 5)), 'kwargs': str({})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    @test.testcase.skip('Static method tracing was disabled due the bug. This test should be skipped until we find the way to address it.')
    def test_static(self, mock_start, mock_stop):
        fake_cls = FakeTraceStaticMethod()
        self.assertEqual(25, fake_cls.static_method(25))
        expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profilerosprofiler.tests.unit.test_profiler.FakeTraceStatic.method4', 'args': str((25,)), 'kwargs': str({})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_static_method_skip(self, mock_start, mock_stop):
        self.assertEqual(25, FakeTraceStaticMethodSkip.static_method(25))
        self.assertFalse(mock_start.called)
        self.assertFalse(mock_stop.called)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_class_method_skip(self, mock_start, mock_stop):
        self.assertEqual('foo', FakeTraceClassMethodSkip.class_method('foo'))
        self.assertFalse(mock_start.called)
        self.assertFalse(mock_stop.called)