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
def _assert_autologging_input_kwargs_are_superset(autologging_call_input, user_call_input):
    assert set(user_call_input.keys()).issubset(set(autologging_call_input.keys())), "Keyword or dictionary arguments to original function omit one or more expected keys: '{}'".format(set(user_call_input.keys()) - set(autologging_call_input.keys()))