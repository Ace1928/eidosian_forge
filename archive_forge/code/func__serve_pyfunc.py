import logging
import multiprocessing
import os
import shutil
import signal
import sys
from pathlib import Path
from subprocess import Popen, check_call
from typing import List
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DEPLOYMENT_FLAVOR_NAME, MLFLOW_DISABLE_ENV_CREATION
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import _extract_conda_env, mlserver, scoring_server
from mlflow.store.artifact.models_artifact_repo import REGISTERED_MODEL_META_FILE_NAME
from mlflow.utils import env_manager as em
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import read_yaml
from mlflow.utils.virtualenv import _get_or_create_virtualenv
from mlflow.version import VERSION as MLFLOW_VERSION
def _serve_pyfunc(model, env_manager):
    disable_nginx = os.getenv(DISABLE_NGINX, 'false').lower() == 'true'
    enable_mlserver = os.getenv(ENABLE_MLSERVER, 'false').lower() == 'true'
    disable_env_creation = MLFLOW_DISABLE_ENV_CREATION.get()
    conf = model.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
        if not disable_env_creation:
            _install_pyfunc_deps(MODEL_PATH, install_mlflow=True, enable_mlserver=enable_mlserver, env_manager=env_manager)
        if env_manager == em.CONDA:
            bash_cmds.append('source /miniconda/bin/activate custom_env')
        elif env_manager == em.VIRTUALENV:
            bash_cmds.append('source /opt/activate')
    procs = []
    start_nginx = True
    if disable_nginx or enable_mlserver:
        start_nginx = False
    if start_nginx:
        nginx_conf = Path(mlflow.models.__file__).parent.joinpath('container', 'scoring_server', 'nginx.conf')
        nginx = Popen(['nginx', '-c', nginx_conf]) if start_nginx else None
        check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
        check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
        procs.append(nginx)
    cpu_count = multiprocessing.cpu_count()
    inference_server_kwargs = {}
    if enable_mlserver:
        inference_server = mlserver
        nworkers = int(os.getenv('MLSERVER_INFER_WORKERS', cpu_count))
        port = DEFAULT_MLSERVER_PORT
        model_meta = _read_registered_model_meta(MODEL_PATH)
        model_dict = model.to_dict()
        inference_server_kwargs = {'model_name': model_meta.get('model_name'), 'model_version': model_meta.get('model_version', model_dict.get('run_id', model_dict.get('model_uuid')))}
    else:
        inference_server = scoring_server
        nworkers = cpu_count
        port = DEFAULT_INFERENCE_SERVER_PORT
    cmd, cmd_env = inference_server.get_cmd(model_uri=MODEL_PATH, nworkers=nworkers, port=port, **inference_server_kwargs)
    bash_cmds.append(cmd)
    inference_server_process = Popen(['/bin/bash', '-c', ' && '.join(bash_cmds)], env=cmd_env)
    procs.append(inference_server_process)
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[p.pid for p in procs]))
    awaited_pids = _await_subprocess_exit_any(procs=procs)
    _sigterm_handler(awaited_pids)