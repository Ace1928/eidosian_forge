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
def _resolve_extra_tags(autologging_integration, extra_tags):
    tags = {MLFLOW_AUTOLOGGING: autologging_integration}
    if extra_tags:
        if isinstance(extra_tags, dict):
            if MLFLOW_AUTOLOGGING in extra_tags:
                extra_tags.pop(MLFLOW_AUTOLOGGING)
                _logger.warning(f'Tag `{MLFLOW_AUTOLOGGING}` is ignored as it is a reserved tag by MLflow autologging.')
            tags.update(extra_tags)
        else:
            raise mlflow.exceptions.MlflowException.invalid_parameter_value(f'Invalid `extra_tags` type: expecting dictionary, received `{type(extra_tags).__name__}`')
    return tags