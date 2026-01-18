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
def _wrap_patch(destination, name, patch_obj, settings=None):
    """Apply a patch.

    Args:
        destination: Patch destination.
        name: Name of the attribute at the destination.
        patch_obj: Patch object, it should be a function or a property decorated function
            to be assigned to the patch point {destination}.{name}.
        settings: Settings for gorilla.Patch.

    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patch = gorilla.Patch(destination, name, patch_obj, settings=settings)
    gorilla.apply(patch)
    return patch