import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def _generate_model_predictions(self, compute_latency=False):
    """
        Helper method for generating model predictions
        """

    def predict_with_latency(X_copy):
        y_pred_list = []
        pred_latencies = []
        if len(X_copy) == 0:
            raise ValueError('Empty input data')
        is_dataframe = isinstance(X_copy, pd.DataFrame)
        for row in X_copy.iterrows() if is_dataframe else enumerate(X_copy):
            i, row_data = row
            single_input = row_data.to_frame().T if is_dataframe else row_data
            start_time = time.time()
            y_pred = self.model.predict(single_input)
            end_time = time.time()
            pred_latencies.append(end_time - start_time)
            y_pred_list.append(y_pred)
        self.metrics_values.update({_LATENCY_METRIC_NAME: MetricValue(scores=pred_latencies)})
        sample_pred = y_pred_list[0]
        if isinstance(sample_pred, pd.DataFrame):
            return pd.concat(y_pred_list)
        elif isinstance(sample_pred, np.ndarray):
            return np.concatenate(y_pred_list, axis=0)
        elif isinstance(sample_pred, list):
            return sum(y_pred_list, [])
        elif isinstance(sample_pred, pd.Series):
            return pd.concat(y_pred_list, ignore_index=True)
        else:
            raise MlflowException(message=f'Unsupported prediction type {type(sample_pred)} for model type {self.model_type}.', error_code=INVALID_PARAMETER_VALUE)
    X_copy = self.X.copy_to_avoid_mutation()
    if self.model is not None:
        _logger.info('Computing model predictions.')
        if compute_latency:
            model_predictions = predict_with_latency(X_copy)
        else:
            model_predictions = self.model.predict(X_copy)
    else:
        if self.dataset.predictions_data is None:
            raise MlflowException(message='Predictions data is missing when model is not provided. Please provide predictions data in a dataset or provide a model. See the documentation for mlflow.evaluate() for how to specify the predictions data in a dataset.', error_code=INVALID_PARAMETER_VALUE)
        if compute_latency:
            _logger.warning('Setting the latency to 0 for all entries because the model is not provided.')
            self.metrics_values.update({_LATENCY_METRIC_NAME: MetricValue(scores=[0.0] * len(X_copy))})
        model_predictions = self.dataset.predictions_data
    if self.model_type == _ModelType.CLASSIFIER:
        self.label_list = np.unique(self.y)
        self.num_classes = len(self.label_list)
        if self.predict_fn is not None:
            self.y_pred = self.predict_fn(self.X.copy_to_avoid_mutation())
        else:
            self.y_pred = self.dataset.predictions_data
        self.is_binomial = self.num_classes <= 2
        if self.is_binomial:
            if self.pos_label in self.label_list:
                self.label_list = np.delete(self.label_list, np.where(self.label_list == self.pos_label))
                self.label_list = np.append(self.label_list, self.pos_label)
            elif self.pos_label is None:
                self.pos_label = self.label_list[-1]
            _logger.info(f'The evaluation dataset is inferred as binary dataset, positive label is {self.label_list[1]}, negative label is {self.label_list[0]}.')
        else:
            _logger.info(f'The evaluation dataset is inferred as multiclass dataset, number of classes is inferred as {self.num_classes}')
        if self.predict_proba_fn is not None:
            self.y_probs = self.predict_proba_fn(self.X.copy_to_avoid_mutation())
        else:
            self.y_probs = None
    output_column_name = self.predictions
    self.y_pred, self.other_output_columns, self.predictions = _extract_output_and_other_columns(model_predictions, output_column_name)
    self.other_output_columns_for_eval = set()