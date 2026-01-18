import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@dataclasses.dataclass
class ConstantArgument:
    value: Union[int, float, bool, None]