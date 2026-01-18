import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
class BackboneType(enum.Enum):
    TIMM = 'timm'
    TRANSFORMERS = 'transformers'