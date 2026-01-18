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
@staticmethod
def _reconcile(replacement: _LoadElement, existing: _LoadElement) -> _LoadElement:
    """define behavior for when two Load objects are to be put into
        the context.attributes under the same key.

        :param replacement: ``_LoadElement`` that seeks to replace the
         existing one

        :param existing: ``_LoadElement`` that is already present.

        """
    if replacement._reconcile_to_other:
        return existing
    elif replacement._reconcile_to_other is False:
        return replacement
    elif existing._reconcile_to_other:
        return replacement
    elif existing._reconcile_to_other is False:
        return existing
    if existing is replacement:
        return replacement
    elif existing.strategy == replacement.strategy and existing.local_opts == replacement.local_opts:
        return replacement
    elif replacement.is_opts_only:
        existing = existing._clone()
        existing.local_opts = existing.local_opts.union(replacement.local_opts)
        existing._extra_criteria += replacement._extra_criteria
        return existing
    elif existing.is_opts_only:
        replacement = replacement._clone()
        replacement.local_opts = replacement.local_opts.union(existing.local_opts)
        replacement._extra_criteria += existing._extra_criteria
        return replacement
    elif replacement.path.is_token:
        return replacement
    raise sa_exc.InvalidRequestError(f'Loader strategies for {replacement.path} conflict')