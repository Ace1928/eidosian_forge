import importlib
import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import ray.util.client_connect
from ray._private.ray_constants import (
from ray._private.utils import check_ray_client_dependencies_installed, split_address
from ray._private.worker import BaseContext
from ray._private.worker import init as ray_driver_init
from ray.job_config import JobConfig
from ray.util.annotations import Deprecated, PublicAPI
def _get_builder_from_address(address: Optional[str]) -> ClientBuilder:
    if address == 'local':
        return _LocalClientBuilder('local')
    if address is None:
        address = ray._private.services.canonicalize_bootstrap_address(address)
        return _LocalClientBuilder(address)
    module_string, inner_address = _split_address(address)
    try:
        module = importlib.import_module(module_string)
    except Exception as e:
        raise RuntimeError(f'Module: {module_string} does not exist.\nThis module was parsed from Address: {address}') from e
    assert 'ClientBuilder' in dir(module), f'Module: {module_string} does not have ClientBuilder.'
    return module.ClientBuilder(inner_address)