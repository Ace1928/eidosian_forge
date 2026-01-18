import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class TraceDecoratorTestCase(test.TestCase):

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_duplicate_trace_disallow(self, mock_start, mock_stop):

        @profiler.trace('test')
        def trace_me():
            pass
        self.assertRaises(ValueError, profiler.trace('test-again', allow_multiple_trace=False), trace_me)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_with_args(self, mock_start, mock_stop):
        self.assertEqual(1, traced_func(1))
        expected_info = {'info': 'some_info', 'function': {'name': 'osprofiler.tests.unit.test_profiler.traced_func', 'args': str((1,)), 'kwargs': str({})}}
        mock_start.assert_called_once_with('function', info=expected_info)
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_without_args(self, mock_start, mock_stop):
        self.assertEqual((1, 2), trace_hide_args_func(1, i=2))
        expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.trace_hide_args_func'}}
        mock_start.assert_called_once_with('hide_args', info=expected_info)
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_with_exception(self, mock_start, mock_stop):
        self.assertRaises(ValueError, test_fn_exc)
        expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.test_fn_exc'}}
        expected_stop_info = {'etype': 'ValueError', 'message': ''}
        mock_start.assert_called_once_with('foo', info=expected_info)
        mock_stop.assert_called_once_with(info=expected_stop_info)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_with_result(self, mock_start, mock_stop):
        self.assertEqual((1, 2), trace_with_result_func(1, i=2))
        start_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.trace_with_result_func', 'args': str((1,)), 'kwargs': str({'i': 2})}}
        stop_info = {'function': {'result': str((1, 2))}}
        mock_start.assert_called_once_with('hide_result', info=start_info)
        mock_stop.assert_called_once_with(info=stop_info)