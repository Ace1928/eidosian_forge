import hashlib
import json
import logging
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from filelock import FileLock
import ray
from ray._private.runtime_env.conda_utils import (
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.validation import parse_and_validate_conda
from ray._private.utils import (
class CondaPlugin(RuntimeEnvPlugin):
    name = 'conda'

    def __init__(self, resources_dir: str):
        self._resources_dir = os.path.join(resources_dir, 'conda')
        try_to_create_directory(self._resources_dir)
        self._installs_and_deletions_file_lock = os.path.join(self._resources_dir, 'ray-conda-installs-and-deletions.lock')
        self._validated_named_conda_env = set()

    def _get_path_from_hash(self, hash: str) -> str:
        """Generate a path from the hash of a conda or pip spec.

        The output path also functions as the name of the conda environment
        when using the `--prefix` option to `conda create` and `conda remove`.

        Example output:
            /tmp/ray/session_2021-11-03_16-33-59_356303_41018/runtime_resources
                /conda/ray-9a7972c3a75f55e976e620484f58410c920db091
        """
        return os.path.join(self._resources_dir, hash)

    def get_uris(self, runtime_env: 'RuntimeEnv') -> List[str]:
        """Return the conda URI from the RuntimeEnv if it exists, else return []."""
        conda_uri = runtime_env.conda_uri()
        if conda_uri:
            return [conda_uri]
        return []

    def delete_uri(self, uri: str, logger: Optional[logging.Logger]=default_logger) -> int:
        """Delete URI and return the number of bytes deleted."""
        logger.info(f'Got request to delete URI {uri}')
        protocol, hash = parse_uri(uri)
        if protocol != Protocol.CONDA:
            raise ValueError(f'CondaPlugin can only delete URIs with protocol conda.  Received protocol {protocol}, URI {uri}')
        conda_env_path = self._get_path_from_hash(hash)
        local_dir_size = get_directory_size_bytes(conda_env_path)
        with FileLock(self._installs_and_deletions_file_lock):
            successful = delete_conda_env(prefix=conda_env_path, logger=logger)
        if not successful:
            logger.warning(f'Error when deleting conda env {conda_env_path}. ')
            return 0
        return local_dir_size

    async def create(self, uri: Optional[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger=default_logger) -> int:
        if not runtime_env.has_conda():
            return 0

        def _create():
            result = parse_and_validate_conda(runtime_env.get('conda'))
            if isinstance(result, str):
                if result in self._validated_named_conda_env:
                    return 0
                conda_env_list = get_conda_env_list()
                envs = [Path(env).name for env in conda_env_list]
                if result not in envs:
                    raise ValueError(f"The given conda environment '{result}' from the runtime env {runtime_env} doesn't exist from the output of `conda env list --json`. You can only specify an env that already exists. Please make sure to create an env {result} ")
                self._validated_named_conda_env.add(result)
                return 0
            logger.debug(f'Setting up conda for runtime_env: {runtime_env.serialize()}')
            protocol, hash = parse_uri(uri)
            conda_env_name = self._get_path_from_hash(hash)
            conda_dict = _get_conda_dict_with_ray_inserted(runtime_env, logger=logger)
            logger.info(f'Setting up conda environment with {runtime_env}')
            with FileLock(self._installs_and_deletions_file_lock):
                try:
                    conda_yaml_file = os.path.join(self._resources_dir, 'environment.yml')
                    with open(conda_yaml_file, 'w') as file:
                        yaml.dump(conda_dict, file)
                    create_conda_env_if_needed(conda_yaml_file, prefix=conda_env_name, logger=logger)
                finally:
                    os.remove(conda_yaml_file)
                if runtime_env.get_extension('_inject_current_ray'):
                    _inject_ray_to_conda_site(conda_path=conda_env_name, logger=logger)
            logger.info(f'Finished creating conda environment at {conda_env_name}')
            return get_directory_size_bytes(conda_env_name)
        loop = get_or_create_event_loop()
        return await loop.run_in_executor(None, _create)

    def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger):
        if not runtime_env.has_conda():
            return
        if runtime_env.conda_env_name():
            conda_env_name = runtime_env.conda_env_name()
        else:
            protocol, hash = parse_uri(runtime_env.conda_uri())
            conda_env_name = self._get_path_from_hash(hash)
        context.py_executable = 'python'
        context.command_prefix += get_conda_activate_commands(conda_env_name)