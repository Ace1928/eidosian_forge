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
def _save_model_metadata(dst_dir, spark_model, mlflow_model, sample_input, conda_env, code_paths, signature=None, input_example=None, pip_requirements=None, extra_pip_requirements=None, remote_model_path=None):
    """
    Saves model metadata into the passed-in directory.
    If mlflowdbfs is not used, the persisted metadata assumes that a model can be
    loaded from a relative path to the metadata file (currently hard-coded to "sparkml").
    If mlflowdbfs is used, remote_model_path should be provided, and the model needs to
    be loaded from the remote_model_path.
    """
    import pyspark
    is_spark_connect_model = _is_spark_connect_model(spark_model)
    if sample_input is not None and (not is_spark_connect_model):
        mleap.add_to_model(mlflow_model=mlflow_model, path=dst_dir, spark_model=spark_model, sample_input=sample_input)
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, dst_dir)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, dst_dir)
    mlflow_model.add_flavor(FLAVOR_NAME, pyspark_version=pyspark.__version__, model_data=_SPARK_MODEL_PATH_SUB, code=code_dir_subpath, model_class=_get_fully_qualified_class_name(spark_model))
    pyfunc.add_to_model(mlflow_model, loader_module='mlflow.spark', data=_SPARK_MODEL_PATH_SUB, conda_env=_CONDA_ENV_FILE_NAME, python_env=_PYTHON_ENV_FILE_NAME, code=code_dir_subpath)
    if (size := get_total_file_size(dst_dir)):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(dst_dir, MLMODEL_FILE_NAME))
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(is_spark_connect_model)
            if remote_model_path:
                _logger.info('Inferring pip requirements by reloading the logged model from the databricks artifact repository, which can be time-consuming. To speed up, explicitly specify the conda_env or pip_requirements when calling log_model().')
            inferred_reqs = mlflow.models.infer_pip_requirements(remote_model_path or dst_dir, FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(default_reqs, pip_requirements, extra_pip_requirements)
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)
    with open(os.path.join(dst_dir, _CONDA_ENV_FILE_NAME), 'w') as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    if pip_constraints:
        write_to(os.path.join(dst_dir, _CONSTRAINTS_FILE_NAME), '\n'.join(pip_constraints))
    write_to(os.path.join(dst_dir, _REQUIREMENTS_FILE_NAME), '\n'.join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(dst_dir, _PYTHON_ENV_FILE_NAME))