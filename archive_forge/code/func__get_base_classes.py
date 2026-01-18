import sys
import dis
from typing import List, Tuple, TypeVar
from types import FunctionType
def _get_base_classes(frame, namespace):
    return [_get_base_class(class_name_components, namespace) for class_name_components in _get_base_class_names(frame)]