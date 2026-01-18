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
def _upload_package(self, package_uri: str, package_path: str, include_parent_dir: Optional[bool]=False, excludes: Optional[List[str]]=None, is_file: bool=False) -> bool:
    logger.info(f'Uploading package {package_uri}.')
    with tempfile.TemporaryDirectory() as tmp_dir:
        protocol, package_name = uri_to_http_components(package_uri)
        if is_file:
            package_file = Path(package_path)
        else:
            package_file = Path(tmp_dir) / package_name
            create_package(package_path, package_file, include_parent_dir=include_parent_dir, excludes=excludes)
        try:
            r = self._do_request('PUT', f'/api/packages/{protocol}/{package_name}', data=package_file.read_bytes())
            if r.status_code != 200:
                self._raise_error(r)
        finally:
            if not is_file:
                package_file.unlink()