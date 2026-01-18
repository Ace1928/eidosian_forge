import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _convert_actor(self, actor: 'ActorClass') -> str:
    """Register a ClientActorClass for the ActorClass and return a UUID"""
    return self.worker._convert_actor(actor)