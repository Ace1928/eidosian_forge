import concurrent.futures
import logging
import sys
import threading
import uuid
from py4j.java_gateway import CallbackServerParameters
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.spark import FLAVOR_NAME
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import _truncate_and_ellipsize
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import get_repl_id as get_databricks_repl_id
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH
def _get_table_info_string(path, version, data_format):
    if data_format == 'delta':
        return f'path={path},version={version},format={data_format}'
    return f'path={path},format={data_format}'