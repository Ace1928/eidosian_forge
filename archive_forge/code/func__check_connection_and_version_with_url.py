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
def _check_connection_and_version_with_url(self, min_version: str='1.9', version_error_message: str=None, url: str='/api/version'):
    if version_error_message is None:
        version_error_message = f'Please ensure the cluster is running Ray {min_version} or higher.'
    try:
        r = self._do_request('GET', url)
        if r.status_code == 404:
            raise RuntimeError('Version check returned 404. ' + version_error_message)
        r.raise_for_status()
        running_ray_version = r.json()['ray_version']
        if packaging.version.parse(running_ray_version) < packaging.version.parse(min_version):
            raise RuntimeError(f'Ray version {running_ray_version} is running on the cluster. ' + version_error_message)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f'Failed to connect to Ray at address: {self._address}.')