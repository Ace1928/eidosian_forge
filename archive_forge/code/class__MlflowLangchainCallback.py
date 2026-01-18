import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
class _MlflowLangchainCallback(MlflowCallbackHandler, metaclass=ExceptionSafeAbstractClass):
    """
        Callback for auto-logging metrics and parameters.
        We need to inherit ExceptionSafeAbstractClass to avoid invalid new
        input arguments added to original function call.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)