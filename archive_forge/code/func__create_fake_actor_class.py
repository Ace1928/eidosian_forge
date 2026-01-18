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
def _create_fake_actor_class(self, actor_class_name, actor_method_names, traceback_str):

    class TemporaryActor:
        pass

    def temporary_actor_method(*args, **kwargs):
        raise RuntimeError(f'The actor with name {actor_class_name} failed to import on the worker. This may be because needed library dependencies are not installed in the worker environment:\n\n{traceback_str}')
    for method in actor_method_names:
        setattr(TemporaryActor, method, temporary_actor_method)
    return TemporaryActor