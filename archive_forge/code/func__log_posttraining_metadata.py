import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _log_posttraining_metadata(estimator, spark_model, params, input_df):
    if _is_parameter_search_estimator(estimator):
        try:
            child_tags = context_registry.resolve_tags()
            child_tags.update({MLFLOW_AUTOLOGGING: AUTOLOGGING_INTEGRATION_NAME})
            _create_child_runs_for_parameter_search(parent_estimator=estimator, parent_model=spark_model, parent_run=mlflow.active_run(), child_tags=child_tags)
        except Exception:
            msg = f'Encountered exception during creation of child runs for parameter search. Child runs may be missing. Exception: {traceback.format_exc()}'
            _logger.warning(msg)
        estimator_param_maps = _get_tuning_param_maps(estimator, estimator._autologging_metadata.uid_to_indexed_name_map)
        metrics_dict, best_index = _get_param_search_metrics_and_best_index(estimator, spark_model)
        _log_parameter_search_results_as_artifact(estimator_param_maps, metrics_dict, mlflow.active_run().info.run_id)
        best_param_map = estimator_param_maps[best_index]
        mlflow.log_dict(best_param_map, artifact_file='best_parameters.json')
        _log_estimator_params({f'best_{param_name}': param_value for param_name, param_value in best_param_map.items()})
    if log_models:
        if _should_log_model(spark_model):
            from pyspark.sql import SparkSession
            from mlflow.models import infer_signature
            from mlflow.pyspark.ml._autolog import cast_spark_df_with_vector_to_array, get_feature_cols
            from mlflow.spark import _find_and_set_features_col_as_vector_if_needed
            spark = SparkSession.builder.getOrCreate()

            def _get_input_example_as_pd_df():
                feature_cols = list(get_feature_cols(input_df, spark_model))
                limited_input_df = input_df.select(feature_cols).limit(INPUT_EXAMPLE_SAMPLE_ROWS)
                return cast_spark_df_with_vector_to_array(limited_input_df).toPandas()

            def _infer_model_signature(input_example_slice):
                input_slice_df = _find_and_set_features_col_as_vector_if_needed(spark.createDataFrame(input_example_slice), spark_model)
                model_output = spark_model.transform(input_slice_df).drop(*input_slice_df.columns)
                unsupported_columns = _get_columns_with_unsupported_data_type(model_output)
                if unsupported_columns:
                    _logger.warning(f'Model outputs contain unsupported Spark data types: {unsupported_columns}. Output schema is not be logged.')
                    model_output = None
                else:
                    model_output = model_output.toPandas()
                return infer_signature(input_example_slice, model_output)
            nonlocal log_model_signatures
            if log_model_signatures:
                unsupported_columns = _get_columns_with_unsupported_data_type(input_df)
                if unsupported_columns:
                    _logger.warning(f'Model inputs contain unsupported Spark data types: {unsupported_columns}. Model signature is not logged.')
                    log_model_signatures = False
            input_example, signature = resolve_input_example_and_signature(_get_input_example_as_pd_df, _infer_model_signature, log_input_examples, log_model_signatures, _logger)
            mlflow.spark.log_model(spark_model, artifact_path='model', registered_model_name=registered_model_name, input_example=input_example, signature=signature)
            if _is_parameter_search_model(spark_model):
                mlflow.spark.log_model(spark_model.bestModel, artifact_path='best_model')
        else:
            _logger.warning(_get_warning_msg_for_skip_log_model(spark_model))