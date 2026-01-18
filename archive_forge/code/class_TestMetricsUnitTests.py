from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import pprint
import six
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from boto.storage_uri import BucketStorageUri
from gslib import metrics
from gslib import VERSION
from gslib.cs_api_map import ApiSelector
import gslib.exception
from gslib.gcs_json_api import GcsJsonApi
from gslib.metrics import MetricsCollector
from gslib.metrics_tuple import Metric
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SkipForParFile
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from six import add_move, MovedModule
from six.moves import mock
@SkipForParFile('Do not try spawning the interpreter nested in the archive.')
@mock.patch('time.time', new=mock.MagicMock(return_value=0))
class TestMetricsUnitTests(testcase.GsUtilUnitTestCase):
    """Unit tests for analytics data collection."""

    def setUp(self):
        super(TestMetricsUnitTests, self).setUp()
        self.original_collector_instance = MetricsCollector.GetCollector()
        MetricsCollector.StartTestCollector('https://example.com', 'user-agent-007', {'a': 'b', 'c': 'd'})
        self.collector = MetricsCollector.GetCollector()
        self.log_handler = MockLoggingHandler()
        logging.getLogger('metrics').setLevel(logging.DEBUG)
        logging.getLogger('metrics').addHandler(self.log_handler)

    def tearDown(self):
        super(TestMetricsUnitTests, self).tearDown()
        MetricsCollector.StopTestCollector(original_instance=self.original_collector_instance)

    def testDisabling(self):
        """Tests enabling/disabling of metrics collection."""
        self.assertEqual(self.collector, MetricsCollector.GetCollector())
        with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': '1', 'GA_CID': '555'}):
            MetricsCollector._CheckAndSetDisabledCache()
            self.assertFalse(MetricsCollector._disabled_cache)
            self.assertEqual(self.collector, MetricsCollector.GetCollector())
        with mock.patch('boto.config.getbool', return_value=True):
            MetricsCollector._CheckAndSetDisabledCache()
            self.assertTrue(MetricsCollector._disabled_cache)
            self.assertEqual(None, MetricsCollector.GetCollector())
        with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': '1', 'GA_CID': ''}):
            MetricsCollector._CheckAndSetDisabledCache()
            self.assertTrue(MetricsCollector._disabled_cache)
            self.assertEqual(None, MetricsCollector.GetCollector())
        with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': ''}):
            with mock.patch('os.path.exists', return_value=False):
                MetricsCollector._CheckAndSetDisabledCache()
                self.assertTrue(MetricsCollector._disabled_cache)
                self.assertEqual(None, MetricsCollector.GetCollector())
        with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': ''}):
            with mock.patch('os.path.exists', return_value=True):
                if six.PY2:
                    builtin_open = '__builtin__.open'
                else:
                    builtin_open = 'builtins.open'
                with mock.patch(builtin_open) as mock_open:
                    mock_open.return_value.__enter__ = lambda s: s
                    mock_open.return_value.read.return_value = metrics._DISABLED_TEXT
                    MetricsCollector._CheckAndSetDisabledCache()
                    self.assertTrue(MetricsCollector._disabled_cache)
                    self.assertEqual(None, MetricsCollector.GetCollector())
                    mock_open.return_value.read.return_value = 'mock_cid'
                    MetricsCollector._CheckAndSetDisabledCache()
                    self.assertFalse(MetricsCollector._disabled_cache)
                    self.assertEqual(self.collector, MetricsCollector.GetCollector())
                    self.assertEqual(2, len(mock_open.call_args_list))
                    self.assertEqual(2, len(mock_open.return_value.read.call_args_list))

    def testConfigValueValidation(self):
        """Tests the validation of potentially PII config values."""
        string_and_bool_categories = ['check_hashes', 'content_language', 'disable_analytics_prompt', 'https_validate_certificates', 'json_api_version', 'parallel_composite_upload_component_size', 'parallel_composite_upload_threshold', 'prefer_api', 'sliced_object_download_component_size', 'sliced_object_download_threshold', 'tab_completion_time_logs', 'token_cache', 'use_magicfile']
        int_categories = ['debug', 'default_api_version', 'http_socket_timeout', 'max_retry_delay', 'num_retries', 'oauth2_refresh_retries', 'parallel_process_count', 'parallel_thread_count', 'resumable_threshold', 'rsync_buffer_lines', 'sliced_object_download_max_components', 'software_update_check_period', 'tab_completion_timeout', 'task_estimation_threshold']
        all_categories = sorted(string_and_bool_categories + int_categories)
        with mock.patch('boto.config.get_value', return_value=None):
            self.assertEqual('', self.collector._ValidateAndGetConfigValues())
        with mock.patch('boto.config.get_value', return_value='invalid string'):
            self.assertEqual(','.join([category + ':INVALID' for category in all_categories]), self.collector._ValidateAndGetConfigValues())
        with mock.patch('boto.config.get_value', return_value='£'):
            self.assertEqual(','.join([category + ':INVALID' for category in all_categories]), self.collector._ValidateAndGetConfigValues())

        def MockValidStrings(section, category):
            if section == 'GSUtil':
                if category == 'check_hashes':
                    return 'if_fast_else_skip'
                if category == 'content_language':
                    return 'chi'
                if category == 'json_api_version':
                    return 'v3'
                if category == 'prefer_api':
                    return 'xml'
                if category in ('disable_analytics_prompt', 'use_magicfile', 'tab_completion_time_logs'):
                    return 'True'
            if section == 'OAuth2' and category == 'token_cache':
                return 'file_system'
            if section == 'Boto' and category == 'https_validate_certificates':
                return 'True'
            return ''
        with mock.patch('boto.config.get_value', side_effect=MockValidStrings):
            self.assertEqual('check_hashes:if_fast_else_skip,content_language:chi,disable_analytics_prompt:True,https_validate_certificates:True,json_api_version:v3,prefer_api:xml,tab_completion_time_logs:True,token_cache:file_system,use_magicfile:True', self.collector._ValidateAndGetConfigValues())

        def MockValidSmallInts(_, category):
            if category in int_categories:
                return '1999'
            return ''
        with mock.patch('boto.config.get_value', side_effect=MockValidSmallInts):
            self.assertEqual('debug:1999,default_api_version:1999,http_socket_timeout:1999,max_retry_delay:1999,num_retries:1999,oauth2_refresh_retries:1999,parallel_process_count:1999,parallel_thread_count:1999,resumable_threshold:1999,rsync_buffer_lines:1999,sliced_object_download_max_components:1999,software_update_check_period:1999,tab_completion_timeout:1999,task_estimation_threshold:1999', self.collector._ValidateAndGetConfigValues())

        def MockValidLargeInts(_, category):
            if category in int_categories:
                return '2001'
            return ''
        with mock.patch('boto.config.get_value', side_effect=MockValidLargeInts):
            self.assertEqual('debug:INVALID,default_api_version:INVALID,http_socket_timeout:INVALID,max_retry_delay:INVALID,num_retries:INVALID,oauth2_refresh_retries:INVALID,parallel_process_count:INVALID,parallel_thread_count:INVALID,resumable_threshold:2001,rsync_buffer_lines:2001,sliced_object_download_max_components:INVALID,software_update_check_period:INVALID,tab_completion_timeout:INVALID,task_estimation_threshold:2001', self.collector._ValidateAndGetConfigValues())

            def MockNonIntegerValue(_, category):
                if category in int_categories:
                    return '10.28'
                return ''
            with mock.patch('boto.config.get_value', side_effect=MockNonIntegerValue):
                self.assertEqual(','.join([category + ':INVALID' for category in int_categories]), self.collector._ValidateAndGetConfigValues())

            def MockDataSizeValue(_, category):
                if category in ('parallel_composite_upload_component_size', 'parallel_composite_upload_threshold', 'sliced_object_download_component_size', 'sliced_object_download_threshold'):
                    return '10MiB'
                return ''
            with mock.patch('boto.config.get_value', side_effect=MockDataSizeValue):
                self.assertEqual('parallel_composite_upload_component_size:10485760,parallel_composite_upload_threshold:10485760,sliced_object_download_component_size:10485760,sliced_object_download_threshold:10485760', self.collector._ValidateAndGetConfigValues())

    def testCommandAndErrorEventsCollection(self):
        """Tests the collection of command and error GA events."""
        self.assertEqual([], self.collector._metrics)
        _LogAllTestMetrics()
        metrics.LogCommandParams(command_name='cmd2')
        self.assertEqual([], self.collector._metrics)
        self.collector._CollectCommandAndErrorMetrics()
        self.assertEqual(COMMAND_AND_ERROR_TEST_METRICS, set(self.collector._metrics))

    def testPerformanceSummaryEventCollection(self):
        """Test the collection of PerformanceSummary GA events."""
        self.collector.ga_params[metrics._GA_LABEL_MAP['Command Name']] = 'cp'
        with mock.patch('gslib.metrics.system_util.GetDiskCounters', return_value={'fake-disk': (0, 0, 0, 0, 0, 0)}):
            metrics.LogPerformanceSummaryParams(uses_fan=True, uses_slice=True, avg_throughput=10, is_daisy_chain=True, has_file_dst=False, has_cloud_dst=True, has_file_src=False, has_cloud_src=True, total_bytes_transferred=100, total_elapsed_time=10, thread_idle_time=40, thread_execution_time=10, num_processes=2, num_threads=3, num_objects_transferred=3, provider_types=['gs'])
        service_retry_msg = RetryableErrorMessage(apitools_exceptions.CommunicationError(), 0)
        network_retry_msg = RetryableErrorMessage(socket.error(), 0)
        metrics.LogRetryableError(service_retry_msg)
        metrics.LogRetryableError(network_retry_msg)
        metrics.LogRetryableError(network_retry_msg)
        start_file_msg = FileMessage('src', 'dst', 0, size=100)
        end_file_msg = FileMessage('src', 'dst', 10, finished=True)
        start_file_msg.thread_id = end_file_msg.thread_id = 1
        start_file_msg.process_id = end_file_msg.process_id = 1
        metrics.LogPerformanceSummaryParams(file_message=start_file_msg)
        metrics.LogPerformanceSummaryParams(file_message=end_file_msg)
        self.assertEqual(self.collector.perf_sum_params.thread_throughputs[1, 1].GetThroughput(), 10)
        with mock.patch('gslib.metrics.system_util.GetDiskCounters', return_value={'fake-disk': (0, 0, 0, 0, 10, 10)}):
            self.collector._CollectPerformanceSummaryMetric()
        metric_body = self.collector._metrics[0].body
        label_and_value_pairs = [('Event Category', metrics._GA_PERFSUM_CATEGORY), ('Event Action', 'CloudToCloud%2CDaisyChain'), ('Execution Time', '10'), ('Parallelism Strategy', 'both'), ('Source URL Type', 'cloud'), ('Provider Types', 'gs'), ('Num Processes', '2'), ('Num Threads', '3'), ('Number of Files/Objects Transferred', '3'), ('Size of Files/Objects Transferred', '100'), ('Average Overall Throughput', '10'), ('Num Retryable Service Errors', '1'), ('Num Retryable Network Errors', '2'), ('Thread Idle Time Percent', '0.8'), ('Slowest Thread Throughput', '10'), ('Fastest Thread Throughput', '10')]
        if IS_LINUX:
            label_and_value_pairs.append(('Disk I/O Time', '20'))
        for label, exp_value in label_and_value_pairs:
            self.assertIn('{0}={1}'.format(metrics._GA_LABEL_MAP[label], exp_value), metric_body)

    def testCommandCollection(self):
        """Tests the collection of command parameters."""
        _TryExceptAndPass(self.command_runner.RunNamedCommand, 'acl', ['set', '-a'], collect_analytics=True)
        self.assertEqual('acl set', self.collector.ga_params.get(metrics._GA_LABEL_MAP['Command Name']))
        self.assertEqual('a', self.collector.ga_params.get(metrics._GA_LABEL_MAP['Command-Level Options']))
        self.collector.ga_params.clear()
        self.command_runner.RunNamedCommand('list', collect_analytics=True)
        self.assertEqual('ls', self.collector.ga_params.get(metrics._GA_LABEL_MAP['Command Name']))
        self.assertEqual('list', self.collector.ga_params.get(metrics._GA_LABEL_MAP['Command Alias']))
        self.collector.ga_params.clear()
        _TryExceptAndPass(self.command_runner.RunNamedCommand, 'iam', ['get', 'dummy_bucket'], collect_analytics=True)
        self.assertEqual('iam get', self.collector.ga_params.get(metrics._GA_LABEL_MAP['Command Name']))

    @mock.patch.object(http_wrapper, 'HandleExceptionsAndRebuildHttpConnections')
    def testRetryableErrorCollection(self, mock_default_retry):
        """Tests the collection of a retryable error in the retry function."""
        mock_queue = RetryableErrorsQueue()
        value_error_retry_args = http_wrapper.ExceptionRetryArgs(None, None, ValueError(), None, None, None)
        socket_error_retry_args = http_wrapper.ExceptionRetryArgs(None, None, socket.error(), None, None, None)
        metadata_retry_func = LogAndHandleRetries(is_data_transfer=False, status_queue=mock_queue)
        media_retry_func = LogAndHandleRetries(is_data_transfer=True, status_queue=mock_queue)
        metadata_retry_func(value_error_retry_args)
        self.assertEqual(self.collector.retryable_errors['ValueError'], 1)
        metadata_retry_func(value_error_retry_args)
        self.assertEqual(self.collector.retryable_errors['ValueError'], 2)
        metadata_retry_func(socket_error_retry_args)
        if six.PY2:
            self.assertEqual(self.collector.retryable_errors['SocketError'], 1)
        else:
            self.assertEqual(self.collector.retryable_errors['OSError'], 1)
        _TryExceptAndPass(media_retry_func, value_error_retry_args)
        _TryExceptAndPass(media_retry_func, socket_error_retry_args)
        self.assertEqual(self.collector.retryable_errors['ValueError'], 3)
        if six.PY2:
            self.assertEqual(self.collector.retryable_errors['SocketError'], 2)
        else:
            self.assertEqual(self.collector.retryable_errors['OSError'], 2)

    def testExceptionCatchingDecorator(self):
        """Tests the exception catching decorator CaptureAndLogException."""
        mock_exc_fn = mock.MagicMock(__name__=str('mock_exc_fn'), side_effect=Exception())
        wrapped_fn = metrics.CaptureAndLogException(mock_exc_fn)
        wrapped_fn()
        debug_messages = self.log_handler.messages['debug']
        self.assertIn('Exception captured in mock_exc_fn during metrics collection', debug_messages[0])
        self.log_handler.reset()
        self.assertEqual(1, mock_exc_fn.call_count)
        mock_err_fn = mock.MagicMock(__name__=str('mock_err_fn'), side_effect=TypeError())
        wrapped_fn = metrics.CaptureAndLogException(mock_err_fn)
        wrapped_fn()
        self.assertEqual(1, mock_err_fn.call_count)
        debug_messages = self.log_handler.messages['debug']
        self.assertIn('Exception captured in mock_err_fn during metrics collection', debug_messages[0])
        self.log_handler.reset()
        with mock.patch.object(MetricsCollector, 'GetCollector', return_value='not a collector'):
            metrics.Shutdown()
            metrics.LogCommandParams()
            metrics.LogRetryableError()
            metrics.LogFatalError()
            metrics.LogPerformanceSummaryParams()
            metrics.CheckAndMaybePromptForAnalyticsEnabling('invalid argument')
            debug_messages = self.log_handler.messages['debug']
            message_index = 0
            for func_name in ('Shutdown', 'LogCommandParams', 'LogRetryableError', 'LogFatalError', 'LogPerformanceSummaryParams', 'CheckAndMaybePromptForAnalyticsEnabling'):
                self.assertIn('Exception captured in %s during metrics collection' % func_name, debug_messages[message_index])
                message_index += 1
            self.log_handler.reset()