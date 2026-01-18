import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _resolve_domain_vars(spec: Dict, domain_vars: List[Tuple[Tuple, Domain]], allow_fail: bool=False, random_state: 'RandomState'=None) -> Tuple[bool, Dict]:
    resolved = {}
    error = True
    num_passes = 0
    while error and num_passes < _MAX_RESOLUTION_PASSES:
        num_passes += 1
        error = False
        for path, domain in domain_vars:
            if path in resolved:
                continue
            try:
                value = domain.sample(_UnresolvedAccessGuard(spec), random_state=random_state)
            except RecursiveDependencyError as e:
                error = e
            except Exception:
                raise ValueError('Failed to evaluate expression: {}: {}'.format(path, domain))
            else:
                assign_value(spec, path, value)
                resolved[path] = value
    if error:
        if not allow_fail:
            raise error
        else:
            return (False, resolved)
    return (True, resolved)