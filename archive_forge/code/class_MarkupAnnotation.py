import sys
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from ..generic import ArrayObject, DictionaryObject
from ..generic._base import (
from ..generic._fit import DEFAULT_FIT, Fit
from ..generic._rectangle import RectangleObject
from ..generic._utils import hex_to_rgb
from ._base import NO_FLAGS, AnnotationDictionary
class MarkupAnnotation(AnnotationDictionary, ABC):
    """
    Base class for all markup annotations.

    Args:
        title_bar: Text to be displayed in the title bar of the annotation;
            by convention this is the name of the author
    """

    def __init__(self, *, title_bar: Optional[str]=None):
        if title_bar is not None:
            self[NameObject('T')] = TextStringObject(title_bar)