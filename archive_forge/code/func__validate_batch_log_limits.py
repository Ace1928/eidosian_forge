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
def _validate_batch_log_limits(metrics, params, tags):
    """Validate that the provided batched logging arguments are within expected limits."""
    _validate_batch_limit(entity_name='metrics', limit=MAX_METRICS_PER_BATCH, length=len(metrics))
    _validate_batch_limit(entity_name='params', limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(params))
    _validate_batch_limit(entity_name='tags', limit=MAX_PARAMS_TAGS_PER_BATCH, length=len(tags))
    total_length = len(metrics) + len(params) + len(tags)
    _validate_batch_limit(entity_name='metrics, params, and tags', limit=MAX_ENTITIES_PER_BATCH, length=total_length)