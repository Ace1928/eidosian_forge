from __future__ import annotations
import abc
import pickle
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from ..util.typing import Self
class DefaultSerialization:
    serializer: Union[None, Serializer] = staticmethod(pickle.dumps)
    deserializer: Union[None, Deserializer] = staticmethod(pickle.loads)