import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _register_callback(self, ref: 'ClientObjectRef', callback: Callable[['DataResponse'], None]) -> None:
    self.worker.register_callback(ref, callback)