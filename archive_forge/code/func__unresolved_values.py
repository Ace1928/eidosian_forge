import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _unresolved_values(spec: Dict) -> Dict[Tuple, Any]:
    return _split_resolved_unresolved_values(spec)[1]