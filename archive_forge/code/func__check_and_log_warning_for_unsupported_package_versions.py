import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def _check_and_log_warning_for_unsupported_package_versions(integration_name):
    """
    When autologging is enabled and `disable_for_unsupported_versions=False` for the specified
    autologging integration, check whether the currently-installed versions of the integration's
    associated package versions are supported by the specified integration. If the package versions
    are not supported, log a warning message.
    """
    if integration_name in FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY and (not get_autologging_config(integration_name, 'disable', True)) and (not get_autologging_config(integration_name, 'disable_for_unsupported_versions', False)) and (not is_flavor_supported_for_associated_package_versions(integration_name)):
        _logger.warning('You are using an unsupported version of %s. If you encounter errors during autologging, try upgrading / downgrading %s to a supported version, or try upgrading MLflow.', integration_name, integration_name)