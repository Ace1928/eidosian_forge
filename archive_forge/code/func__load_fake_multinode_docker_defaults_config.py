import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_fake_multinode_docker_defaults_config():
    import ray.autoscaler._private.fake_multi_node as ray_fake_multinode
    return os.path.join(os.path.dirname(ray_fake_multinode.__file__), 'example_docker.yaml')