import functools
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _log_lightgbm_dataset(lgb_dataset, source, context, autologging_client, name=None):
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse
    data = lgb_dataset.data
    label = lgb_dataset.label
    if isinstance(data, pd.DataFrame):
        dataset = from_pandas(df=data, source=source, name=name)
    elif issparse(data):
        arr_data = data.toarray() if issparse(data) else data
        dataset = from_numpy(features=arr_data, targets=label, source=source, name=name)
    elif isinstance(data, np.ndarray):
        dataset = from_numpy(features=data, targets=label, source=source, name=name)
    else:
        _logger.warning('Unrecognized dataset type %s. Dataset logging skipped.', type(data))
        return
    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=context)]
    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
    autologging_client.log_inputs(run_id=mlflow.active_run().info.run_id, datasets=[dataset_input])