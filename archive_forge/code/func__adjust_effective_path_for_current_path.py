from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
def _adjust_effective_path_for_current_path(self, effective_path: PathRegistry, current_path: PathRegistry) -> Optional[PathRegistry]:
    """receives the 'current_path' entry from an :class:`.ORMCompileState`
        instance, which is set during lazy loads and secondary loader strategy
        loads, and adjusts the given path to be relative to the
        current_path.

        E.g. given a loader path and current path::

            lp: User -> orders -> Order -> items -> Item -> keywords -> Keyword

            cp: User -> orders -> Order -> items

        The adjusted path would be::

            Item -> keywords -> Keyword


        """
    chopped_start_path = Load._chop_path(effective_path.natural_path, current_path)
    if not chopped_start_path:
        return None
    tokens_removed_from_start_path = len(effective_path) - len(chopped_start_path)
    loader_lead_path_element = self.path[tokens_removed_from_start_path]
    effective_path = PathRegistry.coerce((loader_lead_path_element,) + chopped_start_path[1:])
    return effective_path