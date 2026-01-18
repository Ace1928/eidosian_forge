import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
def _bind(self):
    """Bind the args and kwargs to the `placement_group()` signature.

        We bind the args and kwargs, so we can compare equality of two resource
        requests. The main reason for this is that the `placement_group()` API
        can evolve independently from the ResourceRequest API (e.g. adding new
        arguments). Then, `ResourceRequest(bundles, strategy, arg=arg)` should
        be the same as `ResourceRequest(bundles, strategy, arg)`.
        """
    sig = signature(placement_group)
    try:
        self._bound = sig.bind(self._bundles, self._strategy, *self._args, **self._kwargs)
    except Exception as exc:
        raise RuntimeError('Invalid definition for resource request. Please check that you passed valid arguments to the ResourceRequest object.') from exc