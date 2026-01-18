import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from asyncio import create_task, get_running_loop
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.utils import check_output_cmd
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
import ray
class PipProcessor:

    def __init__(self, target_dir: str, runtime_env: 'RuntimeEnv', logger: Optional[logging.Logger]=default_logger):
        try:
            import virtualenv
        except ImportError:
            raise RuntimeError(f'Please install virtualenv `{sys.executable} -m pip install virtualenv`to enable pip runtime env.')
        logger.debug('Setting up pip for runtime_env: %s', runtime_env)
        self._target_dir = target_dir
        self._runtime_env = runtime_env
        self._logger = logger
        self._pip_config = self._runtime_env.pip_config()
        self._pip_env = os.environ.copy()
        self._pip_env.update(self._runtime_env.env_vars())

    @staticmethod
    def _is_in_virtualenv() -> bool:
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    @classmethod
    async def _ensure_pip_version(cls, path: str, pip_version: Optional[str], cwd: str, pip_env: Dict, logger: logging.Logger):
        """Run the pip command to reinstall pip to the specified version."""
        if not pip_version:
            return
        python = _PathHelper.get_virtualenv_python(path)
        pip_reinstall_cmd = [python, '-m', 'pip', 'install', '--disable-pip-version-check', f'pip{pip_version}']
        logger.info('Installing pip with version %s', pip_version)
        await check_output_cmd(pip_reinstall_cmd, logger=logger, cwd=cwd, env=pip_env)

    async def _pip_check(self, path: str, pip_check: bool, cwd: str, pip_env: Dict, logger: logging.Logger):
        """Run the pip check command to check python dependency conflicts.
        If exists conflicts, the exit code of pip check command will be non-zero.
        """
        if not pip_check:
            logger.info('Skip pip check.')
            return
        python = _PathHelper.get_virtualenv_python(path)
        await check_output_cmd([python, '-m', 'pip', 'check', '--disable-pip-version-check'], logger=logger, cwd=cwd, env=pip_env)
        logger.info('Pip check on %s successfully.', path)

    @staticmethod
    @asynccontextmanager
    async def _check_ray(python: str, cwd: str, logger: logging.Logger):
        """A context manager to check ray is not overwritten.

        Currently, we only check ray version and path. It works for virtualenv,
          - ray is in Python's site-packages.
          - ray is overwritten during yield.
          - ray is in virtualenv's site-packages.
        """

        async def _get_ray_version_and_path() -> Tuple[str, str]:
            with tempfile.TemporaryDirectory(prefix='check_ray_version_tempfile') as tmp_dir:
                ray_version_path = os.path.join(tmp_dir, 'ray_version.txt')
                check_ray_cmd = [python, '-c', '\nimport ray\nwith open(r"{ray_version_path}", "wt") as f:\n    f.write(ray.__version__)\n    f.write(" ")\n    f.write(ray.__path__[0])\n                    '.format(ray_version_path=ray_version_path)]
                if _WIN32:
                    env = os.environ.copy()
                else:
                    env = {}
                output = await check_output_cmd(check_ray_cmd, logger=logger, cwd=cwd, env=env)
                logger.info(f'try to write ray version information in: {ray_version_path}')
                with open(ray_version_path, 'rt') as f:
                    output = f.read()
                ray_version, ray_path, *_ = [s.strip() for s in output.split()]
            return (ray_version, ray_path)
        version, path = await _get_ray_version_and_path()
        yield
        actual_version, actual_path = await _get_ray_version_and_path()
        if actual_version != version or actual_path != path:
            raise RuntimeError(f'Changing the ray version is not allowed: \n  current version: {actual_version}, current path: {actual_path}\n  expect version: {version}, expect path: {path}\nPlease ensure the dependencies in the runtime_env pip field do not install a different version of Ray.')

    @classmethod
    async def _create_or_get_virtualenv(cls, path: str, cwd: str, logger: logging.Logger):
        """Create or get a virtualenv from path."""
        python = sys.executable
        virtualenv_path = os.path.join(path, 'virtualenv')
        virtualenv_app_data_path = os.path.join(path, 'virtualenv_app_data')
        if _WIN32:
            current_python_dir = sys.prefix
            env = os.environ.copy()
        else:
            current_python_dir = os.path.abspath(os.path.join(os.path.dirname(python), '..'))
            env = {}
        if cls._is_in_virtualenv():
            clonevirtualenv = os.path.join(os.path.dirname(__file__), '_clonevirtualenv.py')
            create_venv_cmd = [python, clonevirtualenv, current_python_dir, virtualenv_path]
            logger.info('Cloning virtualenv %s to %s', current_python_dir, virtualenv_path)
        else:
            create_venv_cmd = [python, '-m', 'virtualenv', '--app-data', virtualenv_app_data_path, '--reset-app-data', '--no-periodic-update', '--system-site-packages', '--no-download', virtualenv_path]
            logger.info('Creating virtualenv at %s, current python dir %s', virtualenv_path, virtualenv_path)
        await check_output_cmd(create_venv_cmd, logger=logger, cwd=cwd, env=env)

    @classmethod
    async def _install_pip_packages(cls, path: str, pip_packages: List[str], cwd: str, pip_env: Dict, logger: logging.Logger):
        virtualenv_path = _PathHelper.get_virtualenv_path(path)
        python = _PathHelper.get_virtualenv_python(path)
        pip_requirements_file = _PathHelper.get_requirements_file(path, pip_packages)

        def _gen_requirements_txt():
            with open(pip_requirements_file, 'w') as file:
                for line in pip_packages:
                    file.write(line + '\n')
        loop = get_running_loop()
        await loop.run_in_executor(None, _gen_requirements_txt)
        pip_install_cmd = [python, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', pip_requirements_file]
        logger.info('Installing python requirements to %s', virtualenv_path)
        await check_output_cmd(pip_install_cmd, logger=logger, cwd=cwd, env=pip_env)

    async def _run(self):
        path = self._target_dir
        logger = self._logger
        pip_packages = self._pip_config['packages']
        exec_cwd = os.path.join(path, 'exec_cwd')
        os.makedirs(exec_cwd, exist_ok=True)
        try:
            await self._create_or_get_virtualenv(path, exec_cwd, logger)
            python = _PathHelper.get_virtualenv_python(path)
            async with self._check_ray(python, exec_cwd, logger):
                await self._ensure_pip_version(path, self._pip_config.get('pip_version', None), exec_cwd, self._pip_env, logger)
                await self._install_pip_packages(path, pip_packages, exec_cwd, self._pip_env, logger)
                await self._pip_check(path, self._pip_config.get('pip_check', False), exec_cwd, self._pip_env, logger)
        except Exception:
            logger.info('Delete incomplete virtualenv: %s', path)
            shutil.rmtree(path, ignore_errors=True)
            logger.exception('Failed to install pip packages.')
            raise

    def __await__(self):
        return self._run().__await__()