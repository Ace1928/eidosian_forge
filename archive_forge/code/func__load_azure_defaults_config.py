import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_azure_defaults_config():
    import ray.autoscaler.azure as ray_azure
    return os.path.join(os.path.dirname(ray_azure.__file__), 'defaults.yaml')