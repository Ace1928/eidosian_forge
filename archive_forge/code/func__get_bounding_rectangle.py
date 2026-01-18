import sys
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from ..generic import ArrayObject, DictionaryObject
from ..generic._base import (
from ..generic._fit import DEFAULT_FIT, Fit
from ..generic._rectangle import RectangleObject
from ..generic._utils import hex_to_rgb
from ._base import NO_FLAGS, AnnotationDictionary
def _get_bounding_rectangle(vertices: List[Vertex]) -> RectangleObject:
    x_min, y_min = (vertices[0][0], vertices[0][1])
    x_max, y_max = (vertices[0][0], vertices[0][1])
    for x, y in vertices:
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    rect = RectangleObject((x_min, y_min, x_max, y_max))
    return rect