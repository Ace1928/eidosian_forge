import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from packaging.version import Version
import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.environment import (
from mlflow.utils.file_utils import remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements
def _create_virtualenv(local_model_path, python_bin_path, env_dir, python_env, extra_env=None, capture_output=False):
    paths = ('bin', 'activate') if not is_windows() else ('Scripts', 'activate.bat')
    activate_cmd = env_dir.joinpath(*paths)
    activate_cmd = f'source {activate_cmd}' if not is_windows() else str(activate_cmd)
    if env_dir.exists():
        _logger.info('Environment %s already exists', env_dir)
        return activate_cmd
    with remove_on_error(env_dir, onerror=lambda e: _logger.warning('Encountered an unexpected error: %s while creating a virtualenv environment in %s, removing the environment directory...', repr(e), env_dir)):
        _logger.info('Creating a new environment in %s with %s', env_dir, python_bin_path)
        _exec_cmd([sys.executable, '-m', 'virtualenv', '--python', python_bin_path, env_dir], capture_output=capture_output)
        _logger.info('Installing dependencies')
        for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    for model_item in os.listdir(local_model_path):
                        os.symlink(src=os.path.join(local_model_path, model_item), dst=os.path.join(tmpdir, model_item))
                except Exception as e:
                    _logger.warning('Failed to symlink model directory during dependency installation Copying instead. Exception: %s', e)
                    _copy_model_to_writeable_destination(local_model_path, tmpdir)
                tmp_req_file = f'requirements.{uuid.uuid4().hex}.txt'
                Path(tmpdir).joinpath(tmp_req_file).write_text('\n'.join(deps))
                cmd = _join_commands(activate_cmd, f'python -m pip install --quiet -r {tmp_req_file}')
                _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)
    return activate_cmd