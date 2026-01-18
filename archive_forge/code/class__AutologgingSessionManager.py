import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
class _AutologgingSessionManager:
    _session = None

    @classmethod
    @contextmanager
    def start_session(cls, integration):
        try:
            prev_session = cls._session
            if prev_session is None:
                session_id = uuid.uuid4().hex
                cls._session = AutologgingSession(integration, session_id)
            yield cls._session
        finally:
            if prev_session is None:
                cls._end_session()

    @classmethod
    def active_session(cls):
        return cls._session

    @classmethod
    def _end_session(cls):
        cls._session = None