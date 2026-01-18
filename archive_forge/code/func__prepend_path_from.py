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
def _prepend_path_from(self, parent: Load) -> _LoadElement:
    """adjust the path of this :class:`._LoadElement` to be
        a subpath of that of the given parent :class:`_orm.Load` object's
        path.

        This is used by the :meth:`_orm.Load._apply_to_parent` method,
        which is in turn part of the :meth:`_orm.Load.options` method.

        """
    if not any((orm_util._entity_corresponds_to_use_path_impl(elem, self.path.odd_element(0)) for elem in (parent.path.odd_element(-1),) + parent.additional_source_entities)):
        raise sa_exc.ArgumentError(f'Attribute "{self.path[1]}" does not link from element "{parent.path[-1]}".')
    return self._prepend_path(parent.path)