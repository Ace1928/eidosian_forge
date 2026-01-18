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
def _gen_md5_for_arraylike_obj(md5_gen, data):
    """
    Helper method to generate MD5 hash array-like object, the MD5 will calculate over:
     - array length
     - first NUM_SAMPLE_ROWS_FOR_HASH rows content
     - last NUM_SAMPLE_ROWS_FOR_HASH rows content
    """
    len_bytes = _hash_uint64_ndarray_as_bytes(np.array([len(data)], dtype='uint64'))
    md5_gen.update(len_bytes)
    if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
        md5_gen.update(_hash_array_like_obj_as_bytes(data))
    else:
        head_rows = data[:EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]
        tail_rows = data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH:]
        md5_gen.update(_hash_array_like_obj_as_bytes(head_rows))
        md5_gen.update(_hash_array_like_obj_as_bytes(tail_rows))