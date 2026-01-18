import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def _convert_ray_node_option(key, value):
    converted_key = f'--{key.replace('_', '-')}'
    if key in ['system_config', 'resources', 'labels']:
        return f'{converted_key}={json.dumps(value)}'
    if value is None:
        return converted_key
    return f'{converted_key}={str(value)}'