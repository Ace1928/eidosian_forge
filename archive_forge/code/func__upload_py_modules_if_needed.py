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
def _upload_py_modules_if_needed(self, runtime_env: Dict[str, Any]):

    def _upload_fn(module_path, excludes, is_file=False):
        self._upload_package_if_needed(module_path, include_parent_dir=True, excludes=excludes, is_file=is_file)
    upload_py_modules_if_needed(runtime_env, upload_fn=_upload_fn)