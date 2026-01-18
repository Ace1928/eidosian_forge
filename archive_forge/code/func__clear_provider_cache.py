import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _clear_provider_cache():
    global _provider_instances
    _provider_instances = {}