from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
class _MetricsCollector(object):
    """A singleton class to handle metrics reporting."""
    _disabled_cache = None
    _instance = None
    test_group = None

    @staticmethod
    def GetCollectorIfExists():
        return _MetricsCollector._instance

    @staticmethod
    def GetCollector():
        """Returns the singleton _MetricsCollector instance or None if disabled."""
        if _MetricsCollector._IsDisabled():
            return None
        if not _MetricsCollector._instance:
            _MetricsCollector._instance = _MetricsCollector()
        return _MetricsCollector._instance

    @staticmethod
    def ResetCollectorInstance(disable_cache=None):
        """Reset the singleton _MetricsCollector and reinitialize it.

    This should only be used for tests, where we want to collect some metrics
    but not others, and we have to reinitialize the collector with a different
    Google Analytics tracking id.

    Args:
      disable_cache: Metrics collector keeps an internal cache of the disabled
          state of metrics. This controls the value to reinitialize the cache.
          None means we will refresh the cache with the default values.
          True/False forces a specific value.
    """
        _MetricsCollector._disabled_cache = disable_cache
        if _MetricsCollector._IsDisabled():
            _MetricsCollector._instance = None
        else:
            _MetricsCollector._instance = _MetricsCollector()

    @staticmethod
    def _IsDisabled():
        """Returns whether metrics collection should be disabled."""
        if _MetricsCollector._disabled_cache is None:
            if '_ARGCOMPLETE' in os.environ:
                _MetricsCollector._disabled_cache = True
            elif not properties.IsDefaultUniverse():
                _MetricsCollector._disabled_cache = True
            else:
                disabled = properties.VALUES.core.disable_usage_reporting.GetBool()
                if disabled is None:
                    disabled = config.INSTALLATION_CONFIG.disable_usage_reporting
                _MetricsCollector._disabled_cache = disabled
        return _MetricsCollector._disabled_cache

    def __init__(self):
        """Initialize a new MetricsCollector.

    This should only be invoked through the static GetCollector() function or
    the static ResetCollectorInstance() function.
    """
        common_params = CommonParams()
        self._metrics_reporters = [_ClearcutMetricsReporter(common_params)]
        self._timer = _CommandTimer()
        self._metrics = []
        self._action_level = 0
        current_platform = platforms.Platform.Current()
        self._async_popen_args = current_platform.AsyncPopenArgs()
        log.debug('Metrics collector initialized...')

    def IncrementActionLevel(self):
        self._action_level += 1

    def DecrementActionLevel(self):
        self._action_level -= 1

    def RecordTimedEvent(self, name, record_only_on_top_level=False, event_time=None):
        """Records the time when a particular event happened.

    Args:
      name: str, Name of the event.
      record_only_on_top_level: bool, Whether to record only on top level.
      event_time: float, Time when the event happened in secs since epoch.
    """
        if self._action_level == 0 or not record_only_on_top_level:
            self._timer.Event(name, event_time=event_time)

    def RecordRPCDuration(self, duration_in_ms):
        """Records the time when a particular event happened.

    Args:
      duration_in_ms: int, Duration of the RPC in milli seconds.
    """
        self._timer.AddRPCDuration(duration_in_ms)

    def SetTimerContext(self, category, action, label=None, flag_names=None):
        """Sets the context for which the timer is collecting timed events.

    Args:
      category: str, Category of the action being timed.
      action: str, Name of the action being timed.
      label: str, Additional information about the action being timed.
      flag_names: str, Comma separated list of flag names used with the action.
    """
        if category is _COMMANDS_CATEGORY and self._action_level != 0:
            return
        if category is _ERROR_CATEGORY and self._action_level != 0:
            _, action, _, _ = self._timer.GetContext()
        self._timer.SetContext(category, action, label, flag_names)

    def Record(self, event, flag_names=None, error=None, error_extra_info_json=None):
        """Records the given event.

    Args:
      event: _Event, The event to process.
      flag_names: str, Comma separated list of flag names used with the action.
      error: class, The class (not the instance) of the Exception if a user
        tried to run a command that produced an error.
      error_extra_info_json: {str: json-serializable}, A json serializable dict
        of extra info that we want to log with the error. This enables us to
        write queries that can understand the keys and values in this dict.
    """
        for metrics_reporter in self._metrics_reporters:
            metrics_reporter.Record(event, flag_names=flag_names, error=error, error_extra_info_json=error_extra_info_json)

    def CollectMetrics(self):
        for metrics_reporter in self._metrics_reporters:
            http_beacon = metrics_reporter.ToHTTPBeacon(self._timer)
            self.CollectHTTPBeacon(*http_beacon)

    def CollectHTTPBeacon(self, url, method, body, headers):
        """Record a custom event to an arbitrary endpoint.

    Args:
      url: str, The full url of the endpoint to hit.
      method: str, The HTTP method to issue.
      body: str, The body to send with the request.
      headers: {str: str}, A map of headers to values to include in the request.
    """
        self._metrics.append((url, method, body, headers))

    def ReportMetrics(self, wait_for_report=False):
        """Reports the collected metrics using a separate async process."""
        if not self._metrics:
            return
        temp_metrics_file = tempfile.NamedTemporaryFile(delete=False)
        with temp_metrics_file:
            pickle.dump(self._metrics, temp_metrics_file)
            self._metrics = []
        this_file = encoding.Decode(__file__)
        reporting_script_path = os.path.realpath(os.path.join(os.path.dirname(this_file), 'metrics_reporter.py'))
        execution_args = execution_utils.ArgsForPythonTool(reporting_script_path, temp_metrics_file.name)
        execution_args = [encoding.Encode(a) for a in execution_args]
        exec_env = os.environ.copy()
        encoding.SetEncodedValue(exec_env, 'PYTHONPATH', os.pathsep.join(sys.path))
        try:
            p = subprocess.Popen(execution_args, env=exec_env, **self._async_popen_args)
            log.debug('Metrics reporting process started...')
        except OSError:
            log.debug('Metrics reporting process failed to start.')
        if wait_for_report:
            p.communicate()
            log.debug('Metrics reporting process finished.')