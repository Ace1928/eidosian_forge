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
class _LocalClientBuilder(ClientBuilder):

    def connect(self) -> ClientContext:
        """
        Begin a connection to the address passed in via ray.client(...)
        """
        if self._deprecation_warn_enabled:
            self._client_deprecation_warn()
        self._fill_defaults_from_env()
        connection_dict = ray.init(address=self.address, job_config=self._job_config)
        return ClientContext(dashboard_url=connection_dict['webui_url'], python_version='{}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]), ray_version=ray.__version__, ray_commit=ray.__commit__, protocol_version=None, _num_clients=1, _context_to_restore=None)