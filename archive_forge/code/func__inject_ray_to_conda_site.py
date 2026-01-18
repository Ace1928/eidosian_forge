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
def _inject_ray_to_conda_site(conda_path, logger: Optional[logging.Logger]=default_logger):
    """Write the current Ray site package directory to a new site"""
    if _WIN32:
        python_binary = os.path.join(conda_path, 'python')
    else:
        python_binary = os.path.join(conda_path, 'bin/python')
    site_packages_path = subprocess.check_output([python_binary, '-c', "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).decode().strip()
    ray_path = _resolve_current_ray_path()
    logger.warning(f'Injecting {ray_path} to environment site-packages {site_packages_path} because _inject_current_ray flag is on.')
    maybe_ray_dir = os.path.join(site_packages_path, 'ray')
    if os.path.isdir(maybe_ray_dir):
        logger.warning(f'Replacing existing ray installation with {ray_path}')
        shutil.rmtree(maybe_ray_dir)
    with open(os.path.join(site_packages_path, 'ray_shared.pth'), 'w') as f:
        f.write(ray_path)