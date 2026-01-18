from collections.abc import Sequence
from typing import (
import numpy as np
from ._shape import _ShapeLike
from ._char_codes import (
class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]