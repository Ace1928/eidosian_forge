from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from .util import _mapper_property_as_plain_name
from .. import exc as sa_exc
from .. import util
from ..exc import MultipleResultsFound  # noqa
from ..exc import NoResultFound  # noqa
def _safe_cls_name(cls: Type[Any]) -> str:
    cls_name: Optional[str]
    try:
        cls_name = '.'.join((cls.__module__, cls.__name__))
    except AttributeError:
        cls_name = getattr(cls, '__name__', None)
        if cls_name is None:
            cls_name = repr(cls)
    return cls_name