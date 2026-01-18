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
def _package_exists(self, package_uri: str) -> bool:
    protocol, package_name = uri_to_http_components(package_uri)
    r = self._do_request('GET', f'/api/packages/{protocol}/{package_name}')
    if r.status_code == 200:
        logger.debug(f'Package {package_uri} already exists.')
        return True
    elif r.status_code == 404:
        logger.debug(f'Package {package_uri} does not exist.')
        return False
    else:
        self._raise_error(r)