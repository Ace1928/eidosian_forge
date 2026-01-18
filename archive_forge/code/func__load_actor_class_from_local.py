import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def _load_actor_class_from_local(self, actor_creation_function_descriptor):
    """Load actor class from local code."""
    module_name, class_name = (actor_creation_function_descriptor.module_name, actor_creation_function_descriptor.class_name)
    object = self.load_function_or_class_from_local(module_name, class_name)
    if object is not None:
        if isinstance(object, ray.actor.ActorClass):
            return object.__ray_metadata__.modified_class
        else:
            return object
    else:
        return None