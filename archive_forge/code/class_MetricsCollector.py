from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
class MetricsCollector(object):
    """A singleton class to handle metrics reporting to Google Analytics (GA).

  This class is not thread or process-safe, and logging directly to the
  MetricsCollector instance can only be done by a single thread.
  """

    def __init__(self, ga_tid=_GA_TID, endpoint=_GA_ENDPOINT):
        """Initialize a new MetricsCollector.

    This should only be invoked through the GetCollector or StartTestCollector
    functions.

    Args:
      ga_tid: The Google Analytics tracking ID to use for metrics collection.
      endpoint: The URL to send requests to.
    """
        self.start_time = _GetTimeInMillis()
        self.endpoint = endpoint
        self.logger = logging.getLogger()
        cid = MetricsCollector._GetCID()
        use_test_property = boto.config.getbool('GSUtil', 'use_test_GA_property')
        if use_test_property:
            ga_tid = _GA_TID_TESTING
        config_values = self._ValidateAndGetConfigValues()
        is_corp_user = 0
        if _GOOGLE_CORP_HOST_RE.match(socket.gethostname()):
            is_corp_user = 1
        self.ga_params = {'v': '1', 'tid': ga_tid, 'cid': cid, 't': 'event', _GA_LABEL_MAP['Config']: config_values, _GA_LABEL_MAP['Is Google Corp User']: is_corp_user}
        self.user_agent = '{system}/{release}'.format(system=platform.system(), release=platform.release())
        self._metrics = []
        self.retryable_errors = defaultdict(int)
        self.perf_sum_params = None
    _instance = None
    _disabled_cache = None

    def _ValidateAndGetConfigValues(self):
        """Parses the user's config file to aggregate non-PII config values.

    Returns:
      A comma-delimited string of config values explicitly set by the user in
      key:value pairs, sorted alphabetically by key.
    """
        config_values = []
        invalid_value_string = 'INVALID'

        def GetAndValidateConfigValue(section, category, validation_fn):
            try:
                config_value = boto.config.get_value(section, category)
                if config_value and validation_fn(config_value):
                    config_values.append((category, config_value))
                elif config_value:
                    config_values.append((category, invalid_value_string))
            except:
                config_values.append((category, invalid_value_string))
        for section, bool_category in (('Boto', 'https_validate_certificates'), ('GSUtil', 'disable_analytics_prompt'), ('GSUtil', 'use_magicfile'), ('GSUtil', 'tab_completion_time_logs')):
            GetAndValidateConfigValue(section=section, category=bool_category, validation_fn=lambda val: str(val).lower() in ('true', 'false'))
        small_int_threshold = 2000
        for section, small_int_category in (('Boto', 'debug'), ('Boto', 'http_socket_timeout'), ('Boto', 'num_retries'), ('Boto', 'max_retry_delay'), ('GSUtil', 'default_api_version'), ('GSUtil', 'sliced_object_download_max_components'), ('GSUtil', 'parallel_process_count'), ('GSUtil', 'parallel_thread_count'), ('GSUtil', 'software_update_check_period'), ('GSUtil', 'tab_completion_timeout'), ('OAuth2', 'oauth2_refresh_retries')):
            GetAndValidateConfigValue(section=section, category=small_int_category, validation_fn=lambda val: str(val).isdigit() and int(val) < small_int_threshold)
        for section, large_int_category in (('GSUtil', 'resumable_threshold'), ('GSUtil', 'rsync_buffer_lines'), ('GSUtil', 'task_estimation_threshold')):
            GetAndValidateConfigValue(section=section, category=large_int_category, validation_fn=lambda val: str(val).isdigit())
        for section, data_size_category in (('GSUtil', 'parallel_composite_upload_component_size'), ('GSUtil', 'parallel_composite_upload_threshold'), ('GSUtil', 'sliced_object_download_component_size'), ('GSUtil', 'sliced_object_download_threshold')):
            config_value = boto.config.get_value(section, data_size_category)
            if config_value:
                try:
                    size_in_bytes = HumanReadableToBytes(config_value)
                    config_values.append((data_size_category, size_in_bytes))
                except ValueError:
                    config_values.append((data_size_category, invalid_value_string))
        GetAndValidateConfigValue(section='GSUtil', category='check_hashes', validation_fn=lambda val: val in ('if_fast_else_fail', 'if_fast_else_skip', 'always', 'never'))
        GetAndValidateConfigValue(section='GSUtil', category='content_language', validation_fn=lambda val: val.isalpha() and len(val) <= 3)
        GetAndValidateConfigValue(section='GSUtil', category='json_api_version', validation_fn=lambda val: val[0].lower() == 'v' and val[1:].isdigit())
        GetAndValidateConfigValue(section='GSUtil', category='prefer_api', validation_fn=lambda val: val in ('json', 'xml'))
        GetAndValidateConfigValue(section='OAuth2', category='token_cache', validation_fn=lambda val: val in ('file_system', 'in_memory'))
        return ','.join(sorted(['{0}:{1}'.format(config[0], config[1]) for config in config_values]))

    @staticmethod
    def GetCollector(ga_tid=_GA_TID):
        """Returns the singleton MetricsCollector instance or None if disabled."""
        if MetricsCollector.IsDisabled():
            return None
        if not MetricsCollector._instance:
            MetricsCollector._instance = MetricsCollector(ga_tid)
        return MetricsCollector._instance

    @staticmethod
    def IsDisabled():
        """Returns whether metrics collection should be disabled."""
        if MetricsCollector._disabled_cache is None:
            MetricsCollector._CheckAndSetDisabledCache()
        return MetricsCollector._disabled_cache

    @classmethod
    def _CheckAndSetDisabledCache(cls):
        """Sets _disabled_cache based on user opt-in or out."""
        if os.environ.get('GSUTIL_TEST_ANALYTICS') == '1':
            cls._disabled_cache = True
        elif os.environ.get('GSUTIL_TEST_ANALYTICS') == '2':
            cls._disabled_cache = False
            cls.StartTestCollector()
        elif boto.config.getbool('GSUtil', 'use_gcloud_storage', False):
            cls._disabled_cache = True
        elif system_util.InvokedViaCloudSdk():
            cls._disabled_cache = not os.environ.get('GA_CID')
        elif os.path.exists(_UUID_FILE_PATH):
            with open(_UUID_FILE_PATH) as f:
                cls._disabled_cache = f.read() == _DISABLED_TEXT
        else:
            cls._disabled_cache = True

    @classmethod
    def StartTestCollector(cls, endpoint='https://example.com', user_agent='user-agent-007', ga_params=None):
        """Reset the singleton MetricsCollector with testing parameters.

    Should only be used for tests, where we want to change the default
    parameters.

    Args:
      endpoint: str, URL to post to
      user_agent: str, User-Agent string for header.
      ga_params: A list of two-dimensional string tuples to send as parameters.
    """
        if cls.IsDisabled():
            os.environ['GSUTIL_TEST_ANALYTICS'] = '0'
        cls._disabled_cache = False
        cls._instance = cls(_GA_TID_TESTING, endpoint)
        if ga_params is None:
            ga_params = {'a': 'b', 'c': 'd'}
        cls._instance.ga_params = ga_params
        cls._instance.user_agent = user_agent
        if os.environ['GSUTIL_TEST_ANALYTICS'] != '2':
            cls._instance.start_time = 0

    @classmethod
    def StopTestCollector(cls, original_instance=None):
        """Reset the MetricsCollector with default parameters after testing.

    Args:
      original_instance: The original instance of the MetricsCollector so we can
        set the collector back to its original state.
    """
        os.environ['GSUTIL_TEST_ANALYTICS'] = '1'
        cls._disabled_cache = None
        cls._instance = original_instance

    @staticmethod
    def _GetCID():
        """Gets the client id from the UUID file or the SDK opt-in, or returns None.

    Returns:
      str, The hex string of the client id.
    """
        if os.path.exists(_UUID_FILE_PATH):
            with open(_UUID_FILE_PATH) as f:
                cid = f.read()
            if cid:
                return cid
        return os.environ.get('GA_CID')

    def ExtendGAParams(self, new_params):
        """Extends self.ga_params to include new parameters.

    This is only used to record parameters that are sent with every event type,
    such as global and command-level options.

    Args:
      new_params: A dictionary of key-value parameters to send.
    """
        self.ga_params.update(new_params)

    def GetGAParam(self, param_name):
        """Convenience function for getting a ga_param of the collector.

    Args:
      param_name: The descriptive name of the param (e.g. 'Command Name'). Must
        be a key in _GA_LABEL_MAP.

    Returns:
      The GA parameter specified, or None.
    """
        return self.ga_params.get(_GA_LABEL_MAP[param_name])

    def CollectGAMetric(self, category, action, label=VERSION, value=0, execution_time=None, **custom_params):
        """Adds a GA metric with the given parameters to the metrics queue.

    Args:
      category: str, the GA Event category.
      action: str, the GA Event action.
      label: str, the GA Event label.
      value: int, the GA Event value.
      execution_time: int, the execution time to record in ms.
      **custom_params: A dictionary of key, value pairs containing custom
          metrics and dimensions to send with the GA Event.
    """
        params = [('ec', category), ('ea', action), ('el', label), ('ev', value), (_GA_LABEL_MAP['Timestamp'], _GetTimeInMillis())]
        params.extend([(k, v) for k, v in six.iteritems(custom_params) if v is not None])
        params.extend([(k, v) for k, v in six.iteritems(self.ga_params) if v is not None])
        if execution_time is None:
            execution_time = _GetTimeInMillis() - self.start_time
        params.append((_GA_LABEL_MAP['Execution Time'], execution_time))
        data = urllib.parse.urlencode(sorted(params))
        self._metrics.append(Metric(endpoint=self.endpoint, method='POST', body=data, user_agent=self.user_agent))

    class _PeformanceSummaryParams(object):
        """This class contains information to create a PerformanceSummary event."""

        def __init__(self):
            self.num_processes = 0
            self.num_threads = 0
            self.num_retryable_service_errors = 0
            self.num_retryable_network_errors = 0
            self.provider_types = set()
            if system_util.IS_LINUX:
                self.disk_counters_start = system_util.GetDiskCounters()
            self.uses_fan = False
            self.uses_slice = False
            self.thread_idle_time = 0
            self.thread_execution_time = 0
            self.thread_throughputs = defaultdict(self._ThreadThroughputInformation)
            self.avg_throughput = None
            self.total_elapsed_time = None
            self.total_bytes_transferred = None
            self.num_objects_transferred = 0
            self.is_daisy_chain = False
            self.has_file_dst = False
            self.has_cloud_dst = False
            self.has_file_src = False
            self.has_cloud_src = False

        class _ThreadThroughputInformation(object):
            """A class to keep track of throughput information for a single thread."""

            def __init__(self):
                self.total_bytes_transferred = 0
                self.total_elapsed_time = 0
                self.task_start_time = None
                self.task_size = None

            def LogTaskStart(self, start_time, bytes_to_transfer):
                self.task_start_time = start_time
                self.task_size = bytes_to_transfer

            def LogTaskEnd(self, end_time):
                self.total_elapsed_time += end_time - self.task_start_time
                self.total_bytes_transferred += self.task_size
                self.task_start_time = None
                self.task_size = None

            def GetThroughput(self):
                return CalculateThroughput(self.total_bytes_transferred, self.total_elapsed_time)

    def UpdatePerformanceSummaryParams(self, params):
        """Updates the _PeformanceSummaryParams object.

    Args:
      params: A dictionary of keyword arguments.
        - uses_fan: True if the command uses fan parallelism.
        - uses_slice: True if the command uses slice parallelism.
        - avg_throughput: The average throughput of the data transfer.
        - is_daisy_chain: True if the transfer uses the daisy-chain method.
        - has_file_dst: True if the transfer's destination is a file URL.
        - has_cloud_dst: True if the transfer's destination is in the cloud.
        - has_file_src: True if the transfer has a file URL as a source.
        - has_cloud_src: True if the transfer has a cloud URL as a source.
        - total_elapsed_time: The total amount of time spent on Apply.
        - total_bytes_transferred: The total number of bytes transferred.
        - thread_idle_time: The additional amount of time that threads spent
                            idle in Apply.
        - thread_execution_time: The additional amount of time that threads
                                 spent executing in Apply.
        - num_retryable_service_errors: The additional number of retryable
                                        service errors that occurred.
        - num_retryable_network_errors: The additional number of retryable
                                        network errors that occurred.
        - num_processes: The number of processes used in a call to Apply.
        - num_threads: The number of threads used in a call to Apply.
        - num_objects_transferred: The total number of objects transferred, as
                                   specified by a ProducerThreadMessage.
        - provider_types: A list of additional provider types used.
        - file_message: A FileMessage used to calculate thread throughput and
                        number of objects transferred in the non-parallel case.
    """
        if self.GetGAParam('Command Name') not in ('cp', 'rsync'):
            return
        if self.perf_sum_params is None:
            self.perf_sum_params = self._PeformanceSummaryParams()
        if 'file_message' in params:
            self._ProcessFileMessage(file_message=params['file_message'])
            return
        for param_name, param in six.iteritems(params):
            if param_name in ('uses_fan', 'uses_slice', 'avg_throughput', 'is_daisy_chain', 'has_file_dst', 'has_cloud_dst', 'has_file_src', 'has_cloud_src', 'total_elapsed_time', 'total_bytes_transferred', 'num_objects_transferred'):
                cur_value = getattr(self.perf_sum_params, param_name)
                if not cur_value:
                    setattr(self.perf_sum_params, param_name, param)
            if param_name in ('thread_idle_time', 'thread_execution_time', 'num_retryable_service_errors', 'num_retryable_network_errors'):
                cur_value = getattr(self.perf_sum_params, param_name)
                setattr(self.perf_sum_params, param_name, cur_value + param)
            if param_name in ('num_processes', 'num_threads'):
                cur_value = getattr(self.perf_sum_params, param_name)
                if cur_value < param:
                    setattr(self.perf_sum_params, param_name, param)
            if param_name == 'provider_types':
                self.perf_sum_params.provider_types.update(param)

    def _ProcessFileMessage(self, file_message):
        """Processes FileMessages for thread throughput calculations.

    Update a thread's throughput based on the FileMessage, which marks the start
    or end of a file or component transfer. The FileMessage provides the number
    of bytes transferred as well as start and end time.

    Args:
      file_message: The FileMessage to process.
    """
        thread_info = self.perf_sum_params.thread_throughputs[file_message.process_id, file_message.thread_id]
        if file_message.finished:
            if not (self.perf_sum_params.uses_slice or self.perf_sum_params.uses_fan):
                self.perf_sum_params.num_objects_transferred += 1
            thread_info.LogTaskEnd(file_message.time)
        else:
            thread_info.LogTaskStart(file_message.time, file_message.size)

    def _CollectCommandAndErrorMetrics(self):
        """Aggregates command and error info and adds them to the metrics list."""
        command_name = self.GetGAParam('Command Name')
        if command_name:
            self.CollectGAMetric(category=_GA_COMMANDS_CATEGORY, action=command_name, **{_GA_LABEL_MAP['Retryable Errors']: sum(self.retryable_errors.values())})
        for error_type, num_errors in six.iteritems(self.retryable_errors):
            self.CollectGAMetric(category=_GA_ERRORRETRY_CATEGORY, action=error_type, **{_GA_LABEL_MAP['Retryable Errors']: num_errors})
        fatal_error_type = self.GetGAParam('Fatal Error')
        if fatal_error_type:
            self.CollectGAMetric(category=_GA_ERRORFATAL_CATEGORY, action=fatal_error_type)

    def _CollectPerformanceSummaryMetric(self):
        """Aggregates PerformanceSummary info and adds the metric to the list."""
        if self.perf_sum_params is None:
            return
        custom_params = {}
        for attr_name, label in (('num_processes', 'Num Processes'), ('num_threads', 'Num Threads'), ('num_retryable_service_errors', 'Num Retryable Service Errors'), ('num_retryable_network_errors', 'Num Retryable Network Errors'), ('avg_throughput', 'Average Overall Throughput'), ('num_objects_transferred', 'Number of Files/Objects Transferred'), ('total_bytes_transferred', 'Size of Files/Objects Transferred')):
            custom_params[_GA_LABEL_MAP[label]] = getattr(self.perf_sum_params, attr_name)
        if system_util.IS_LINUX:
            disk_start = self.perf_sum_params.disk_counters_start
            disk_end = system_util.GetDiskCounters()
            custom_params[_GA_LABEL_MAP['Disk I/O Time']] = sum([stat[4] + stat[5] for stat in disk_end.values()]) - sum([stat[4] + stat[5] for stat in disk_start.values()])
        if self.perf_sum_params.has_cloud_src:
            src_url_type = 'both' if self.perf_sum_params.has_file_src else 'cloud'
        else:
            src_url_type = 'file'
        custom_params[_GA_LABEL_MAP['Source URL Type']] = src_url_type
        if self.perf_sum_params.uses_fan:
            strategy = 'both' if self.perf_sum_params.uses_slice else 'fan'
        else:
            strategy = 'slice' if self.perf_sum_params.uses_slice else 'none'
        custom_params[_GA_LABEL_MAP['Parallelism Strategy']] = strategy
        total_time = self.perf_sum_params.thread_idle_time + self.perf_sum_params.thread_execution_time
        if total_time:
            custom_params[_GA_LABEL_MAP['Thread Idle Time Percent']] = float(self.perf_sum_params.thread_idle_time) / float(total_time)
        if self.perf_sum_params.thread_throughputs:
            throughputs = [thread.GetThroughput() for thread in self.perf_sum_params.thread_throughputs.values()]
            custom_params[_GA_LABEL_MAP['Slowest Thread Throughput']] = min(throughputs)
            custom_params[_GA_LABEL_MAP['Fastest Thread Throughput']] = max(throughputs)
        custom_params[_GA_LABEL_MAP['Provider Types']] = ','.join(sorted(self.perf_sum_params.provider_types))
        transfer_types = {'CloudToCloud': self.perf_sum_params.has_cloud_src and self.perf_sum_params.has_cloud_dst, 'CloudToFile': self.perf_sum_params.has_cloud_src and self.perf_sum_params.has_file_dst, 'DaisyChain': self.perf_sum_params.is_daisy_chain, 'FileToCloud': self.perf_sum_params.has_file_src and self.perf_sum_params.has_cloud_dst, 'FileToFile': self.perf_sum_params.has_file_src and self.perf_sum_params.has_file_dst}
        action = ','.join(sorted([transfer_type for transfer_type, cond in six.iteritems(transfer_types) if cond]))
        apply_execution_time = _GetTimeInMillis(self.perf_sum_params.total_elapsed_time)
        self.CollectGAMetric(category=_GA_PERFSUM_CATEGORY, action=action, execution_time=apply_execution_time, **custom_params)

    def ReportMetrics(self, wait_for_report=False, log_level=None, log_file_path=None):
        """Reports the collected metrics using a separate async process.

    Args:
      wait_for_report: bool, True if the main process should wait for the
        subprocess to exit for testing purposes.
      log_level: int, The subprocess logger's level of debugging for testing
        purposes.
      log_file_path: str, The file that the metrics_reporter module should
        write its logs to. If not supplied, the metrics_reporter module will
        use a predetermined default path. This parameter is intended for use
        by tests that need to evaluate the contents of the file at this path.
    """
        self._CollectCommandAndErrorMetrics()
        self._CollectPerformanceSummaryMetric()
        if not self._metrics:
            return
        if not log_level:
            log_level = self.logger.getEffectiveLevel()
        if os.environ.get('GSUTIL_TEST_ANALYTICS') == '2':
            log_level = logging.WARN
        temp_metrics_file = tempfile.NamedTemporaryFile(delete=False)
        temp_metrics_file_name = six.ensure_str(temp_metrics_file.name)
        with temp_metrics_file:
            pickle.dump(self._metrics, temp_metrics_file)
        logging.debug(self._metrics)
        self._metrics = []
        if log_file_path is not None:
            log_file_path = six.ensure_str('r"%s"' % log_file_path)
        reporting_code = six.ensure_str('from gslib.metrics_reporter import ReportMetrics; ReportMetrics(r"{0}", {1}, log_file_path={2})'.format(temp_metrics_file_name, log_level, log_file_path))
        execution_args = [sys.executable, '-c', reporting_code]
        exec_env = os.environ.copy()
        exec_env['PYTHONPATH'] = os.pathsep.join(sys.path)
        sm_env = dict()
        for k, v in six.iteritems(exec_env):
            sm_env[six.ensure_str(k)] = six.ensure_str(v)
        try:
            p = subprocess.Popen(execution_args, env=sm_env, shell=six.PY3 and system_util.IS_WINDOWS)
            self.logger.debug('Metrics reporting process started...')
            if wait_for_report:
                p.communicate()
                self.logger.debug('Metrics reporting process finished.')
        except OSError:
            self.logger.debug('Metrics reporting process failed to start.')
            try:
                os.unlink(temp_metrics_file.name)
            except:
                pass