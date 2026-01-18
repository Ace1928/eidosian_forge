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
def _generate_cluster_metadata():
    """Return a dictionary of cluster metadata."""
    ray_version, python_version = ray._private.utils.compute_version_info()
    metadata = {'ray_version': ray_version, 'python_version': python_version}
    if usage_stats_enabled():
        metadata.update({'schema_version': usage_constant.SCHEMA_VERSION, 'source': os.getenv('RAY_USAGE_STATS_SOURCE', 'OSS'), 'session_id': str(uuid.uuid4()), 'git_commit': ray.__commit__, 'os': sys.platform, 'session_start_timestamp_ms': int(time.time() * 1000)})
        if sys.platform == 'linux':
            lib, ver = platform.libc_ver()
            if not lib:
                metadata.update({'libc_version': 'NA'})
            else:
                metadata.update({'libc_version': f'{lib}:{ver}'})
    return metadata