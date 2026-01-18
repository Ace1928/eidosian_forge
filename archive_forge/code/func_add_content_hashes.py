import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def add_content_hashes(path, allow_non_existing_paths: bool=False):

    def add_hash_of_file(fpath):
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(2 ** 20), b''):
                contents_hasher.update(chunk)
    path = os.path.expanduser(path)
    if allow_non_existing_paths and (not os.path.exists(path)):
        return
    if os.path.isdir(path):
        dirs = []
        for dirpath, _, filenames in os.walk(path):
            dirs.append((dirpath, sorted(filenames)))
        for dirpath, filenames in sorted(dirs):
            contents_hasher.update(dirpath.encode('utf-8'))
            for name in filenames:
                contents_hasher.update(name.encode('utf-8'))
                fpath = os.path.join(dirpath, name)
                add_hash_of_file(fpath)
    else:
        add_hash_of_file(path)