import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
class EvaluationDataset:
    """
    An input dataset for model evaluation. This is intended for use with the
    :py:func:`mlflow.models.evaluate()`
    API.
    """
    NUM_SAMPLE_ROWS_FOR_HASH = 5
    SPARK_DATAFRAME_LIMIT = 10000

    def __init__(self, data, *, targets=None, name=None, path=None, feature_names=None, predictions=None):
        """
        The values of the constructor arguments comes from the `evaluate` call.
        """
        if name is not None and '"' in name:
            raise MlflowException(message=f'Dataset name cannot include a double quote (") but got {name}', error_code=INVALID_PARAMETER_VALUE)
        if path is not None and '"' in path:
            raise MlflowException(message=f'Dataset path cannot include a double quote (") but got {path}', error_code=INVALID_PARAMETER_VALUE)
        self._user_specified_name = name
        self._path = path
        self._hash = None
        self._supported_dataframe_types = (pd.DataFrame,)
        self._spark_df_type = None
        self._labels_data = None
        self._targets_name = None
        self._has_targets = False
        self._predictions_data = None
        self._predictions_name = None
        self._has_predictions = False
        try:
            if 'pyspark' in sys.modules:
                from pyspark.sql import DataFrame as SparkDataFrame
                self._supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
                self._spark_df_type = SparkDataFrame
        except ImportError:
            pass
        if feature_names is not None and len(set(feature_names)) < len(list(feature_names)):
            raise MlflowException(message='`feature_names` argument must be a list containing unique feature names.', error_code=INVALID_PARAMETER_VALUE)
        has_targets = targets is not None
        if has_targets:
            self._has_targets = True
        if isinstance(data, (np.ndarray, list)):
            if has_targets and (not isinstance(targets, (np.ndarray, list))):
                raise MlflowException(message='If data is a numpy array or list of evaluation features, `targets` argument must be a numpy array or list of evaluation labels.', error_code=INVALID_PARAMETER_VALUE)
            shape_message = 'If the `data` argument is a numpy array, it must be a 2-dimensional array, with the second dimension representing the number of features. If the `data` argument is a list, each of its elements must be a feature array of the numpy array or list, and all elements must have the same length.'
            if isinstance(data, list):
                try:
                    data = np.array(data)
                except ValueError as e:
                    raise MlflowException(message=shape_message, error_code=INVALID_PARAMETER_VALUE) from e
            if len(data.shape) != 2:
                raise MlflowException(message=shape_message, error_code=INVALID_PARAMETER_VALUE)
            self._features_data = data
            if has_targets:
                self._labels_data = targets if isinstance(targets, np.ndarray) else np.array(targets)
                if len(self._features_data) != len(self._labels_data):
                    raise MlflowException(message='The input features example rows must be the same length with labels array.', error_code=INVALID_PARAMETER_VALUE)
            num_features = data.shape[1]
            if feature_names is not None:
                feature_names = list(feature_names)
                if num_features != len(feature_names):
                    raise MlflowException(message='feature name list must be the same length with feature data.', error_code=INVALID_PARAMETER_VALUE)
                self._feature_names = feature_names
            else:
                self._feature_names = [f'feature_{str(i + 1).zfill(math.ceil(math.log10(num_features + 1)))}' for i in range(num_features)]
        elif isinstance(data, self._supported_dataframe_types):
            if has_targets and (not isinstance(targets, str)):
                raise MlflowException(message='If data is a Pandas DataFrame or Spark DataFrame, `targets` argument must be the name of the column which contains evaluation labels in the `data` dataframe.', error_code=INVALID_PARAMETER_VALUE)
            if self._spark_df_type and isinstance(data, self._spark_df_type):
                if data.count() > EvaluationDataset.SPARK_DATAFRAME_LIMIT:
                    _logger.warning(f'Specified Spark DataFrame is too large for model evaluation. Only the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows will be used. If you want evaluate on the whole spark dataframe, please manually call `spark_dataframe.toPandas()`.')
                data = data.limit(EvaluationDataset.SPARK_DATAFRAME_LIMIT).toPandas()
            if has_targets:
                self._labels_data = data[targets].to_numpy()
                self._targets_name = targets
            self._has_predictions = predictions is not None
            if self._has_predictions:
                self._predictions_data = data[predictions].to_numpy()
                self._predictions_name = predictions
            if feature_names is not None:
                self._features_data = data[list(feature_names)]
                self._feature_names = feature_names
            else:
                features_data = data
                if has_targets:
                    features_data = features_data.drop(targets, axis=1, inplace=False)
                if self._has_predictions:
                    features_data = features_data.drop(predictions, axis=1, inplace=False)
                self._features_data = features_data
                self._feature_names = [generate_feature_name_if_not_string(c) for c in self._features_data.columns]
        else:
            raise MlflowException(message='The data argument must be a numpy array, a list or a Pandas DataFrame, or spark DataFrame if pyspark package installed.', error_code=INVALID_PARAMETER_VALUE)
        md5_gen = insecure_hash.md5()
        _gen_md5_for_arraylike_obj(md5_gen, self._features_data)
        if self._labels_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._labels_data)
        if self._predictions_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._predictions_data)
        md5_gen.update(','.join(list(map(str, self._feature_names))).encode('UTF-8'))
        self._hash = md5_gen.hexdigest()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def features_data(self):
        """
        return features data as a numpy array or a pandas DataFrame.
        """
        return self._features_data

    @property
    def labels_data(self):
        """
        return labels data as a numpy array
        """
        return self._labels_data

    @property
    def has_targets(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_targets

    @property
    def targets_name(self):
        """
        return targets name
        """
        return self._targets_name

    @property
    def predictions_data(self):
        """
        return labels data as a numpy array
        """
        return self._predictions_data

    @property
    def has_predictions(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_predictions

    @property
    def predictions_name(self):
        """
        return predictions name
        """
        return self._predictions_name

    @property
    def name(self):
        """
        Dataset name, which is specified dataset name or the dataset hash if user don't specify
        name.
        """
        return self._user_specified_name if self._user_specified_name is not None else self.hash

    @property
    def path(self):
        """
        Dataset path
        """
        return self._path

    @property
    def hash(self):
        """
        Dataset hash, includes hash on first 20 rows and last 20 rows.
        """
        return self._hash

    @property
    def _metadata(self):
        """
        Return dataset metadata containing name, hash, and optional path.
        """
        metadata = {'name': self.name, 'hash': self.hash}
        if self.path is not None:
            metadata['path'] = self.path
        return metadata

    def _log_dataset_tag(self, client, run_id, model_uuid):
        """
        Log dataset metadata as a tag "mlflow.datasets", if the tag already exists, it will
        append current dataset metadata into existing tag content.
        """
        existing_dataset_metadata_str = client.get_run(run_id).data.tags.get('mlflow.datasets', '[]')
        dataset_metadata_list = json.loads(existing_dataset_metadata_str)
        for metadata in dataset_metadata_list:
            if metadata['hash'] == self.hash and metadata['name'] == self.name and (metadata['model'] == model_uuid):
                break
        else:
            dataset_metadata_list.append({**self._metadata, 'model': model_uuid})
        dataset_metadata_str = json.dumps(dataset_metadata_list, separators=(',', ':'))
        client.log_batch(run_id, tags=[RunTag('mlflow.datasets', dataset_metadata_str)])

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        if not isinstance(other, EvaluationDataset):
            return False
        if isinstance(self._features_data, np.ndarray):
            is_features_data_equal = np.array_equal(self._features_data, other._features_data)
        else:
            is_features_data_equal = self._features_data.equals(other._features_data)
        return is_features_data_equal and np.array_equal(self._labels_data, other._labels_data) and (self.name == other.name) and (self.path == other.path) and (self._feature_names == other._feature_names)