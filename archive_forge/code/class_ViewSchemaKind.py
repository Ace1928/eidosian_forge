import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
class ViewSchemaKind(Enum):
    aliasing = auto()
    aliasing_inplace = auto()
    non_aliasing = auto()