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
def _generate_extra_criteria(self, context):
    """Apply the current bound parameters in a QueryContext to the
        immediate "extra_criteria" stored with this Load object.

        Load objects are typically pulled from the cached version of
        the statement from a QueryContext.  The statement currently being
        executed will have new values (and keys) for bound parameters in the
        extra criteria which need to be applied by loader strategies when
        they handle this criteria for a result set.

        """
    assert self._extra_criteria, 'this should only be called if _extra_criteria is present'
    orig_query = context.compile_state.select_statement
    current_query = context.query
    k1 = orig_query._generate_cache_key()
    k2 = current_query._generate_cache_key()
    return k2._apply_params_to_element(k1, and_(*self._extra_criteria))