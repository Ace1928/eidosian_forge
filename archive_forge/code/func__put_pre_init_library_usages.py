import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def _put_pre_init_library_usages():
    assert _internal_kv_initialized()
    if not (ray._private.worker.global_worker.mode == ray.SCRIPT_MODE or ray.util.client.ray.is_connected()):
        return
    for library_usage in _recorded_library_usages:
        _put_library_usage(library_usage)