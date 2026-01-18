import inspect
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import cloudpickle
import yaml
import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _extract_type_hints
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _save_model_with_class_artifacts_params(path, python_model, signature=None, hints=None, artifacts=None, conda_env=None, code_paths=None, mlflow_model=None, pip_requirements=None, extra_pip_requirements=None, model_config=None):
    """
    Args:
        path: The path to which to save the Python model.
        python_model: An instance of a subclass of :class:`~PythonModel`. ``python_model``
            defines how the model loads artifacts and how it performs inference.
        artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
            are resolved to absolute filesystem paths, producing a dictionary of
            ``<name, absolute_path>`` entries, (e.g. {"file": "aboslute_path"}).
            ``python_model`` can reference these resolved entries as the ``artifacts`` property
            of the ``context`` attribute. If ``<artifact_name, 'hf:/repo_id'>``(e.g.
            {"bert-tiny-model": "hf:/prajjwal1/bert-tiny"}) is provided, then the model can be
            fetched from huggingface hub using repo_id `prajjwal1/bert-tiny` directly. If ``None``,
            no artifacts are added to the model.
        conda_env: Either a dictionary representation of a Conda environment or the path to a Conda
            environment yaml file. If provided, this decsribes the environment this model should be
            run in. At minimum, it should specify the dependencies contained in
            :func:`get_default_conda_env()`. If ``None``, the default
            :func:`get_default_conda_env()` environment is added to the model.
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system path before the
            model is loaded.
        mlflow_model: The model to which to add the ``mlflow.pyfunc`` flavor.
        model_config: The model configuration for the flavor. Model configuration is available
            during model loading time.

            .. Note:: Experimental: This parameter may change or be removed in a future release
                without warning.
    """
    if mlflow_model is None:
        mlflow_model = Model()
    custom_model_config_kwargs = {CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__}
    if callable(python_model):
        python_model = _FunctionPythonModel(python_model, hints, signature)
    saved_python_model_subpath = _SAVED_PYTHON_MODEL_SUBPATH
    try:
        with open(os.path.join(path, saved_python_model_subpath), 'wb') as out:
            cloudpickle.dump(python_model, out)
    except Exception as e:
        if 'cannot pickle' in str(e).lower():
            raise MlflowException(f"Failed to serialize Python model. Please audit your class variables (e.g. in `__init__()`) for any unpicklable objects. If you're trying to save an external model in your custom pyfunc, Please use the `artifacts` parameter in `mlflow.pyfunc.save_model()`, and load your external model in the `load_context()` method instead. For example:\n\nclass MyModel(mlflow.pyfunc.PythonModel):\n    def load_context(self, context):\n        model_path = context.artifacts['my_model_path']\n        // custom load logic here\n        self.model = load_model(model_path)\n\nFor more information, see our full tutorial at: https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html\n\nFull serialization error: {e}") from None
        else:
            raise e
    custom_model_config_kwargs[CONFIG_KEY_PYTHON_MODEL] = saved_python_model_subpath
    if artifacts:
        saved_artifacts_config = {}
        with TempDir() as tmp_artifacts_dir:
            saved_artifacts_dir_subpath = 'artifacts'
            hf_prefix = 'hf:/'
            for artifact_name, artifact_uri in artifacts.items():
                if artifact_uri.startswith(hf_prefix):
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError as e:
                        raise MlflowException(f'Failed to import huggingface_hub. Please install huggingface_hub to log the model with artifact_uri {artifact_uri}. Error: {e}')
                    repo_id = artifact_uri[len(hf_prefix):]
                    try:
                        snapshot_location = snapshot_download(repo_id=repo_id, local_dir=os.path.join(path, saved_artifacts_dir_subpath, artifact_name), local_dir_use_symlinks=False)
                    except Exception as e:
                        raise MlflowException.invalid_parameter_value(f'Failed to download snapshot from Hugging Face Hub with artifact_uri: {artifact_uri}. Error: {e}')
                    saved_artifact_subpath = Path(snapshot_location).relative_to(Path(os.path.realpath(path))).as_posix()
                else:
                    tmp_artifact_path = _download_artifact_from_uri(artifact_uri=artifact_uri, output_path=tmp_artifacts_dir.path())
                    relative_path = Path(tmp_artifact_path).relative_to(Path(tmp_artifacts_dir.path())).as_posix()
                    saved_artifact_subpath = os.path.join(saved_artifacts_dir_subpath, relative_path)
                saved_artifacts_config[artifact_name] = {CONFIG_KEY_ARTIFACT_RELATIVE_PATH: saved_artifact_subpath, CONFIG_KEY_ARTIFACT_URI: artifact_uri}
            shutil.move(tmp_artifacts_dir.path(), os.path.join(path, saved_artifacts_dir_subpath))
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config
    saved_code_subpath = _validate_and_copy_code_paths(code_paths, path)
    mlflow.pyfunc.add_to_model(model=mlflow_model, loader_module=_get_pyfunc_loader_module(python_model), code=saved_code_subpath, conda_env=_CONDA_ENV_FILE_NAME, python_env=_PYTHON_ENV_FILE_NAME, model_config=model_config, **custom_model_config_kwargs)
    if (size := get_total_file_size(path)):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(path, mlflow.pyfunc.FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(default_reqs, pip_requirements, extra_pip_requirements)
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)
    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), 'w') as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), '\n'.join(pip_constraints))
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), '\n'.join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))