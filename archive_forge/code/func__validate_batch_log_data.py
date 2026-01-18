import logging
import numbers
import posixpath
import re
from typing import List
from mlflow.entities import Dataset, DatasetInput, InputTag, Param, RunTag
from mlflow.environment_variables import MLFLOW_TRUNCATE_LONG_VALUES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.string_utils import is_string_type
def _validate_batch_log_data(metrics, params, tags):
    for metric in metrics:
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
    return (metrics, [_validate_param(p.key, p.value) for p in params], [_validate_tag(t.key, t.value) for t in tags])