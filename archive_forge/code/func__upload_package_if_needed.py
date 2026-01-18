import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
def _upload_package_if_needed(self, package_path: str, include_parent_dir: bool=False, excludes: Optional[List[str]]=None, is_file: bool=False) -> str:
    if is_file:
        package_uri = get_uri_for_package(Path(package_path))
    else:
        package_uri = get_uri_for_directory(package_path, excludes=excludes)
    if not self._package_exists(package_uri):
        self._upload_package(package_uri, package_path, include_parent_dir=include_parent_dir, excludes=excludes, is_file=is_file)
    else:
        logger.info(f'Package {package_uri} already exists, skipping upload.')
    return package_uri