from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def _setup_entity_query(compile_state, mapper, query_entity, path, adapter, column_collection, with_polymorphic=None, only_load_props=None, polymorphic_discriminator=None, **kw):
    if with_polymorphic:
        poly_properties = mapper._iterate_polymorphic_properties(with_polymorphic)
    else:
        poly_properties = mapper._polymorphic_properties
    quick_populators = {}
    path.set(compile_state.attributes, 'memoized_setups', quick_populators)
    check_for_adapt = adapter and len(path) == 1 and path[-1].is_aliased_class
    for value in poly_properties:
        if only_load_props and value.key not in only_load_props:
            continue
        value.setup(compile_state, query_entity, path, adapter, only_load_props=only_load_props, column_collection=column_collection, memoized_populators=quick_populators, check_for_adapt=check_for_adapt, **kw)
    if polymorphic_discriminator is not None and polymorphic_discriminator is not mapper.polymorphic_on:
        if adapter:
            pd = adapter.columns[polymorphic_discriminator]
        else:
            pd = polymorphic_discriminator
        column_collection.append(pd)