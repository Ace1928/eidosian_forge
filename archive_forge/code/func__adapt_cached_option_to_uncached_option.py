from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
def _adapt_cached_option_to_uncached_option(self, context: QueryContext, uncached_opt: ORMOption) -> ORMOption:
    """adapt this option to the "uncached" version of itself in a
        loader strategy context.

        given "self" which is an option from a cached query, as well as the
        corresponding option from the uncached version of the same query,
        return the option we should use in a new query, in the context of a
        loader strategy being asked to load related rows on behalf of that
        cached query, which is assumed to be building a new query based on
        entities passed to us from the cached query.

        Currently this routine chooses between "self" and "uncached" without
        manufacturing anything new. If the option is itself a loader strategy
        option which has a path, that path needs to match to the entities being
        passed to us by the cached query, so the :class:`_orm.Load` subclass
        overrides this to return "self". For all other options, we return the
        uncached form which may have changing state, such as a
        with_loader_criteria() option which will very often have new state.

        This routine could in the future involve
        generating a new option based on both inputs if use cases arise,
        such as if with_loader_criteria() needed to match up to
        ``AliasedClass`` instances given in the parent query.

        However, longer term it might be better to restructure things such that
        ``AliasedClass`` entities are always matched up on their cache key,
        instead of identity, in things like paths and such, so that this whole
        issue of "the uncached option does not match the entities" goes away.
        However this would make ``PathRegistry`` more complicated and difficult
        to debug as well as potentially less performant in that it would be
        hashing enormous cache keys rather than a simple AliasedInsp. UNLESS,
        we could get cache keys overall to be reliably hashed into something
        like an md5 key.

        .. versionadded:: 1.4.41

        """
    if uncached_opt is not None:
        return uncached_opt
    else:
        return self