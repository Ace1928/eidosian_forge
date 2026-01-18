import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from asyncio import create_task, get_running_loop
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.utils import check_output_cmd
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
import ray
def _get_pip_hash(pip_dict: Dict) -> str:
    serialized_pip_spec = json.dumps(pip_dict, sort_keys=True)
    hash = hashlib.sha1(serialized_pip_spec.encode('utf-8')).hexdigest()
    return hash