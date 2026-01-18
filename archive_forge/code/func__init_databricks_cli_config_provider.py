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
def _init_databricks_cli_config_provider(entry_point):
    """
    set a custom DatabricksConfigProvider with the hostname and token of the
    user running the current command (achieved by looking at
    PythonAccessibleThreadLocals.commandContext, via the already-exposed
    NotebookUtils.getContext API)
    """
    notebook_utils = entry_point.getDbutils().notebook()
    dbr_version = get_databricks_runtime_major_minor_version()
    dbr_major_minor_version = (dbr_version.major, dbr_version.minor)
    if dbr_version.is_client_image or dbr_major_minor_version >= (13, 2):

        class DynamicConfigProvider(DatabricksConfigProvider):

            def get_config(self):
                logger = entry_point.getLogger()
                try:
                    from dbruntime.databricks_repl_context import get_context
                    ctx = get_context()
                    if ctx and ctx.apiUrl and ctx.apiToken:
                        return DatabricksConfig.from_token(host=ctx.apiUrl, token=ctx.apiToken, insecure=ctx.sslTrustAll)
                except Exception as e:
                    print(f'Unexpected internal error while constructing `DatabricksConfig` from REPL context: {e}', file=stderr)
                api_url_option = notebook_utils.getContext().apiUrl()
                api_url = api_url_option.get() if api_url_option.isDefined() else None
                api_token = None
                try:
                    api_token = entry_point.getNonUcApiToken()
                except Exception:
                    fallback_api_token_option = notebook_utils.getContext().apiToken()
                    logger.logUsage('refreshableTokenNotFound', {'api_url': api_url}, None)
                    if fallback_api_token_option.isDefined():
                        api_token = fallback_api_token_option.get()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()
                if api_token is None or api_url is None:
                    return DefaultConfigProvider().get_config()
                return DatabricksConfig.from_token(host=api_url, token=api_token, insecure=ssl_trust_all)
    elif dbr_major_minor_version >= (10, 3):

        class DynamicConfigProvider(DatabricksConfigProvider):

            def get_config(self):
                try:
                    from dbruntime.databricks_repl_context import get_context
                    ctx = get_context()
                    if ctx and ctx.apiUrl and ctx.apiToken:
                        return DatabricksConfig.from_token(host=ctx.apiUrl, token=ctx.apiToken, insecure=ctx.sslTrustAll)
                except Exception as e:
                    print(f'Unexpected internal error while constructing `DatabricksConfig` from REPL context: {e}', file=stderr)
                api_token_option = notebook_utils.getContext().apiToken()
                api_url_option = notebook_utils.getContext().apiUrl()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()
                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return DefaultConfigProvider().get_config()
                return DatabricksConfig.from_token(host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all)
    else:

        class DynamicConfigProvider(DatabricksConfigProvider):

            def get_config(self):
                api_token_option = notebook_utils.getContext().apiToken()
                api_url_option = notebook_utils.getContext().apiUrl()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()
                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return DefaultConfigProvider().get_config()
                return DatabricksConfig.from_token(host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all)
    set_config_provider(DynamicConfigProvider())