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
def _set_of_type_info(self, context, current_path):
    assert self._path_with_polymorphic_path
    pwpi = self._of_type
    assert pwpi
    if not pwpi.is_aliased_class:
        pwpi = inspect(orm_util.AliasedInsp._with_polymorphic_factory(pwpi.mapper.base_mapper, (pwpi.mapper,), aliased=True, _use_mapper_path=True))
    start_path = self._path_with_polymorphic_path
    if current_path:
        new_path = self._adjust_effective_path_for_current_path(start_path, current_path)
        if new_path is None:
            return
        start_path = new_path
    key = ('path_with_polymorphic', start_path.natural_path)
    if key in context:
        existing_aliased_insp = context[key]
        this_aliased_insp = pwpi
        new_aliased_insp = existing_aliased_insp._merge_with(this_aliased_insp)
        context[key] = new_aliased_insp
    else:
        context[key] = pwpi