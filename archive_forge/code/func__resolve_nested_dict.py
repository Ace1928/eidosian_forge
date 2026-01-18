import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _resolve_nested_dict(nested_dict: Dict) -> Dict[Tuple, Any]:
    """Flattens a nested dict by joining keys into tuple of paths.

    Can then be passed into `format_vars`.
    """
    res = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            for k_, v_ in _resolve_nested_dict(v).items():
                res[(k,) + k_] = v_
        else:
            res[k,] = v
    return res