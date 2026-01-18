import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _clean_value(value: Any) -> str:
    """Format floats and replace invalid string characters with ``_``."""
    if isinstance(value, float):
        return f'{value:.4f}'
    else:
        invalid_alphabet = '[^a-zA-Z0-9_-]+'
        return re.sub(invalid_alphabet, '_', str(value)).strip('_')