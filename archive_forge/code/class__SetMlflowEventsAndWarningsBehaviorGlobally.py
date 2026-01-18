import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
class _SetMlflowEventsAndWarningsBehaviorGlobally:
    _lock = RLock()
    _disable_event_logs_count = 0
    _disable_warnings_count = 0
    _reroute_warnings_count = 0

    def __init__(self, disable_event_logs, disable_warnings, reroute_warnings):
        self._disable_event_logs = disable_event_logs
        self._disable_warnings = disable_warnings
        self._reroute_warnings = reroute_warnings

    def __enter__(self):
        try:
            with _SetMlflowEventsAndWarningsBehaviorGlobally._lock:
                if self._disable_event_logs:
                    if _SetMlflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count <= 0:
                        logging_utils.disable_logging()
                    _SetMlflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count += 1
                if self._disable_warnings:
                    if _SetMlflowEventsAndWarningsBehaviorGlobally._disable_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(disabled=True)
                    _SetMlflowEventsAndWarningsBehaviorGlobally._disable_warnings_count += 1
                if self._reroute_warnings:
                    if _SetMlflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                        _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(rerouted=True)
                    _SetMlflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count += 1
        except Exception:
            pass

    def __exit__(self, *args, **kwargs):
        try:
            with _SetMlflowEventsAndWarningsBehaviorGlobally._lock:
                if self._disable_event_logs:
                    _SetMlflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count -= 1
                if self._disable_warnings:
                    _SetMlflowEventsAndWarningsBehaviorGlobally._disable_warnings_count -= 1
                if self._reroute_warnings:
                    _SetMlflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count -= 1
                if _SetMlflowEventsAndWarningsBehaviorGlobally._disable_event_logs_count <= 0:
                    logging_utils.enable_logging()
                if _SetMlflowEventsAndWarningsBehaviorGlobally._disable_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_disablement_state_globally(disabled=False)
                if _SetMlflowEventsAndWarningsBehaviorGlobally._reroute_warnings_count <= 0:
                    _WARNINGS_CONTROLLER.set_mlflow_warnings_rerouting_state_globally(rerouted=False)
        except Exception:
            pass