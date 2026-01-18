import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _try_resolve(v) -> Tuple[bool, Any]:
    if isinstance(v, Domain):
        return (False, v)
    elif isinstance(v, dict) and len(v) == 1 and ('eval' in v):
        return (False, Function(lambda spec: eval(v['eval'], _STANDARD_IMPORTS, {'spec': spec})))
    elif isinstance(v, dict) and len(v) == 1 and ('grid_search' in v):
        grid_values = v['grid_search']
        return (False, Categorical(grid_values).grid())
    return (True, v)