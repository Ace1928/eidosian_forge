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
class _PyFuncModelWrapper:
    """
    Wrapper around Spark MLlib PipelineModel providing interface for scoring pandas DataFrame.
    """

    def __init__(self, spark, spark_model):
        self.spark = spark
        self.spark_model = spark_model

    def predict(self, pandas_df, params: Optional[Dict[str, Any]]=None):
        """
        Generate predictions given input data in a pandas DataFrame.

        Args:
            pandas_df: pandas DataFrame containing input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                                        release without warning.

        Returns:
            List with model predictions.
        """
        from pyspark.ml import PipelineModel
        if _is_spark_connect_model(self.spark_model):
            pandas_df = pandas_df.copy(deep=False)
            return self.spark_model.transform(pandas_df)['prediction']
        spark_df = _find_and_set_features_col_as_vector_if_needed(self.spark.createDataFrame(pandas_df), self.spark_model)
        prediction_column = 'prediction'
        if isinstance(self.spark_model, PipelineModel) and self.spark_model.stages[-1].hasParam('outputCol'):
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            transformed_df = self.spark_model.transform(spark.createDataFrame([], spark_df.schema))
            if prediction_column not in transformed_df.columns:
                self.spark_model.stages[-1].setOutputCol(prediction_column)
        return [x.prediction for x in self.spark_model.transform(spark_df).select(prediction_column).collect()]