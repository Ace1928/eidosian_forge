import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.working_dir import set_pythonpath_in_context
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
class PyModulesPlugin(RuntimeEnvPlugin):
    name = 'py_modules'

    def __init__(self, resources_dir: str, gcs_aio_client: 'GcsAioClient'):
        self._resources_dir = os.path.join(resources_dir, 'py_modules_files')
        self._gcs_aio_client = gcs_aio_client
        try_to_create_directory(self._resources_dir)

    def _get_local_dir_from_uri(self, uri: str):
        return get_local_dir_from_uri(uri, self._resources_dir)

    def delete_uri(self, uri: str, logger: Optional[logging.Logger]=default_logger) -> int:
        """Delete URI and return the number of bytes deleted."""
        logger.info('Got request to delete pymodule URI %s', uri)
        local_dir = get_local_dir_from_uri(uri, self._resources_dir)
        local_dir_size = get_directory_size_bytes(local_dir)
        deleted = delete_package(uri, self._resources_dir)
        if not deleted:
            logger.warning(f'Tried to delete nonexistent URI: {uri}.')
            return 0
        return local_dir_size

    def get_uris(self, runtime_env: dict) -> List[str]:
        return runtime_env.py_modules()

    async def create(self, uri: str, runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger) -> int:
        module_dir = await download_and_unpack_package(uri, self._resources_dir, self._gcs_aio_client, logger=logger)
        if is_whl_uri(uri):
            wheel_uri = module_dir
            module_dir = self._get_local_dir_from_uri(uri)
            await install_wheel_package(wheel_uri=wheel_uri, target_dir=module_dir, logger=logger)
        return get_directory_size_bytes(module_dir)

    def modify_context(self, uris: List[str], runtime_env_dict: Dict, context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger):
        module_dirs = []
        for uri in uris:
            module_dir = self._get_local_dir_from_uri(uri)
            if not module_dir.exists():
                raise ValueError(f'Local directory {module_dir} for URI {uri} does not exist on the cluster. Something may have gone wrong while downloading, unpacking or installing the py_modules files.')
            module_dirs.append(str(module_dir))
        set_pythonpath_in_context(os.pathsep.join(module_dirs), context)