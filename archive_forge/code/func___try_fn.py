import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
def __try_fn(self, domain: 'Function', config: Dict[str, Any]):
    try:
        return domain.func(config)
    except (AttributeError, KeyError):
        from ray.tune.search.variant_generator import _UnresolvedAccessGuard
        r = domain.func(_UnresolvedAccessGuard({'config': config}))
        logger.warning('sample_from functions that take a spec dict are deprecated. Please update your function to work with the config dict directly.')
        return r