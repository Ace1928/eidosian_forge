import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union, overload
from .._fields import MISSING_NONPROP
@dataclasses.dataclass(frozen=True)
class _SubcommandConfiguration:
    name: Optional[str]
    default: Any
    description: Optional[str]
    prefix_name: bool
    constructor_factory: Optional[Callable[[], Union[Type, Callable]]]

    def __hash__(self) -> int:
        return object.__hash__(self)