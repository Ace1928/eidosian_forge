import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path, generate_tmp_dfs_path
def _load_model_fit_in_spark(local_model_path: str, flavor_conf, **kwargs):
    """
    Loads a Diviner model that has been fit (and saved) in the Spark variant.
    """
    import diviner
    dfs_temp_directory = generate_tmp_dfs_path(kwargs.get('dfs_tmpdir', MLFLOW_DFS_TMP.get()))
    dfs_fuse_directory = dbfs_hdfs_uri_to_fuse_path(dfs_temp_directory)
    os.makedirs(dfs_fuse_directory)
    shutil_copytree_without_file_permissions(src_dir=local_model_path, dst_dir=dfs_fuse_directory)
    diviner_instance = getattr(diviner, flavor_conf[_MODEL_TYPE_KEY])
    load_directory = os.path.join(dfs_fuse_directory, flavor_conf[_MODEL_BINARY_KEY])
    return diviner_instance.load(load_directory)