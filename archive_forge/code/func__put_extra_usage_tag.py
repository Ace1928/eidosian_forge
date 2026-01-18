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
def _put_extra_usage_tag(key: str, value: str):
    assert _internal_kv_initialized()
    try:
        _internal_kv_put(f'{usage_constant.EXTRA_USAGE_TAG_PREFIX}{key}'.encode(), value.encode(), namespace=usage_constant.USAGE_STATS_NAMESPACE.encode())
    except Exception as e:
        logger.debug(f'Failed to put extra usage tag, {e}')