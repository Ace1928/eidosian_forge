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
def _patch_implementation(self, original, *args, **kwargs):
    if not mlflow.active_run():
        self.managed_run = create_managed_run()
    result = super()._patch_implementation(original, *args, **kwargs)
    if self.managed_run:
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
    return result