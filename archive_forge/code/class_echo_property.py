from __future__ import annotations
import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from .util import py311
from .util import py38
from .util.typing import Literal
class echo_property:
    __doc__ = "    When ``True``, enable log output for this element.\n\n    This has the effect of setting the Python logging level for the namespace\n    of this element's class and object reference.  A value of boolean ``True``\n    indicates that the loglevel ``logging.INFO`` will be set for the logger,\n    whereas the string value ``debug`` will set the loglevel to\n    ``logging.DEBUG``.\n    "

    @overload
    def __get__(self, instance: Literal[None], owner: Type[Identified]) -> echo_property:
        ...

    @overload
    def __get__(self, instance: Identified, owner: Type[Identified]) -> _EchoFlagType:
        ...

    def __get__(self, instance: Optional[Identified], owner: Type[Identified]) -> Union[echo_property, _EchoFlagType]:
        if instance is None:
            return self
        else:
            return instance._echo

    def __set__(self, instance: Identified, value: _EchoFlagType) -> None:
        instance_logger(instance, echoflag=value)