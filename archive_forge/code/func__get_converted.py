import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _get_converted(self, key: str) -> 'ClientStub':
    """Given a UUID, return the converted object"""
    return self.worker._get_converted(key)