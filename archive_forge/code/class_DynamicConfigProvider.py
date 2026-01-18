import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
class DynamicConfigProvider(DatabricksConfigProvider):

    def get_config(self):
        api_token_option = notebook_utils.getContext().apiToken()
        api_url_option = notebook_utils.getContext().apiUrl()
        ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()
        if not api_token_option.isDefined() or not api_url_option.isDefined():
            return DefaultConfigProvider().get_config()
        return DatabricksConfig.from_token(host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all)