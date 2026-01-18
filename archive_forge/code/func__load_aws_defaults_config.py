import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_aws_defaults_config():
    import ray.autoscaler.aws as ray_aws
    return os.path.join(os.path.dirname(ray_aws.__file__), 'defaults.yaml')