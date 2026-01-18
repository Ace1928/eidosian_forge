import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def _read_sparse_matrix_from_json(path_or_data, example_type):
    if isinstance(path_or_data, str) and os.path.exists(path_or_data):
        with open(path_or_data) as handle:
            matrix_data = json.load(handle)
    else:
        matrix_data = path_or_data
    data = matrix_data['data']
    indices = matrix_data['indices']
    indptr = matrix_data['indptr']
    shape = tuple(matrix_data['shape'])
    if example_type == 'sparse_matrix_csc':
        return csc_matrix((data, indices, indptr), shape=shape)
    else:
        return csr_matrix((data, indices, indptr), shape=shape)