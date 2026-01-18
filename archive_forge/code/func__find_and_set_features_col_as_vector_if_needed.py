import logging
import os
import posixpath
import re
import shutil
from typing import Any, Dict, Optional
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import environment_variables, mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE
from mlflow.models.utils import _Example, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
from mlflow.utils import _get_fully_qualified_class_name, databricks_utils
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
def _find_and_set_features_col_as_vector_if_needed(spark_df, spark_model):
    """
    Finds the `featuresCol` column in spark_model and
    then tries to cast that column to `vector` type.
    This method is noop if the `featuresCol` is already of type `vector`
    or if it can't be cast to `vector` type
    Note:
    If a spark ML pipeline contains a single Estimator stage, it requires
    the input dataframe to contain features column of vector type.
    But the autologging for pyspark ML casts vector column to array<double> type
    for parity with the pd Dataframe. The following fix is required, which transforms
    that features column back to vector type so that the pipeline stages can correctly work.
    A valid scenario is if the auto-logged input example is directly used
    for prediction, which would otherwise fail without this transformation.

    Args:
        spark_df: Input dataframe that contains `featuresCol`
        spark_model: A pipeline model or a single transformer that contains `featuresCol` param

    Returns:
        A spark dataframe that contains features column of `vector` type.
    """
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql import types as t
    from pyspark.sql.functions import udf

    def _find_stage_with_features_col(stage):
        if stage.hasParam('featuresCol'):

            def _array_to_vector(input_array):
                return Vectors.dense(input_array)
            array_to_vector_udf = udf(f=_array_to_vector, returnType=VectorUDT())
            features_col_name = stage.extractParamMap().get(stage.featuresCol)
            features_col_type = [_field for _field in spark_df.schema.fields if _field.name == features_col_name and _field.dataType in [t.ArrayType(t.DoubleType(), True), t.ArrayType(t.DoubleType(), False)]]
            if len(features_col_type) == 1:
                return spark_df.withColumn(features_col_name, array_to_vector_udf(features_col_name))
        return spark_df
    if hasattr(spark_model, 'stages'):
        for stage in reversed(spark_model.stages):
            return _find_stage_with_features_col(stage)
    return _find_stage_with_features_col(spark_model)