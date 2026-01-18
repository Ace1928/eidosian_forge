from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from ..util.typing import Literal
A map that creates new keys for missing key access.

    Produces an incrementing sequence given a series of unique keys.

    This is similar to the compiler prefix_anon_map class although simpler.

    Inlines the approach taken by :class:`sqlalchemy.util.PopulateDict` which
    is otherwise usually used for this type of operation.

    