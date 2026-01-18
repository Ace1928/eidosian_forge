import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, TypeVar, Union
from ray.util.annotations import Deprecated

    Unflatten `flat_key` and iteratively look up in `lookup`. E.g.
    `flat_key="a/0/b"` will try to return `lookup["a"][0]["b"]`.
    