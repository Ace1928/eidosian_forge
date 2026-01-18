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
class SparkAutologgingContext(RunContextProvider):
    """
    Context provider used when there's no active run. Accumulates datasource read information,
    then logs that information to the next-created run. Note that this doesn't clear the accumlated
    info when logging them to the next run, so it will be logged to any successive runs as well.
    """

    def in_context(self):
        return True

    def tags(self):
        if autologging_is_disabled(FLAVOR_NAME):
            return {}
        with _lock:
            global _table_infos
            seen = set()
            unique_infos = []
            for info in _table_infos:
                if info not in seen:
                    unique_infos.append(info)
                    seen.add(info)
            if len(unique_infos) > 0:
                tags = {_SPARK_TABLE_INFO_TAG_NAME: _generate_datasource_tag_value('\n'.join([_get_table_info_string(*info) for info in unique_infos]))}
            else:
                tags = {}
            return tags