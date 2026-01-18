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
def _is_arg_exempt_from_validation(autologging_integration, function_name, argument, argument_index=None, argument_name=None):
    """This function is responsible for determining whether or not an argument is exempt from
    autolog safety validations. This includes both type checking and immutable checking.

    Args:
        autologging_integration: The name of the autologging integration.
        function_name: The name of the function that is being validated.
        argument: The actual argument.
        argument_index: The index of the argument, if it is passed as a positional
            argument. Otherwise it is None.
        argument_name: The name of the argument, if it is passed as a keyword argument.
            Otherwise it is None.

    Returns:
        True or False
    """
    return any((exemption.matches(autologging_integration, function_name, argument, argument_index, argument_name) for exemption in _VALIDATION_EXEMPT_ARGUMENTS))