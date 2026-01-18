import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _flatten_resolved_vars(resolved_vars: Dict) -> Dict:
    """Formats the resolved variable dict into a mapping of (str -> value)."""
    flattened_resolved_vars_dict = {}
    for pieces, value in resolved_vars.items():
        if pieces[0] == 'config':
            pieces = pieces[1:]
        pieces = [str(piece) for piece in pieces]
        flattened_resolved_vars_dict['/'.join(pieces)] = value
    return flattened_resolved_vars_dict