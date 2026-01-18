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
def _get_spark_major_version(sc):
    spark_version_parts = sc.version.split('.')
    spark_major_version = None
    if len(spark_version_parts) > 0:
        spark_major_version = int(spark_version_parts[0])
    return spark_major_version