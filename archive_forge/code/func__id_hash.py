from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
def _id_hash(path_tuple):
    """Compute a hash for the specific placeholder based on its path."""
    return hashlib.sha1(str(path_tuple).encode('utf-8')).hexdigest()[:ID_HASH_LENGTH]