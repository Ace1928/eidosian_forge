from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
class default_compile_options(CacheableOptions):
    _cache_key_traversal = [('_use_legacy_query_style', InternalTraversal.dp_boolean), ('_for_statement', InternalTraversal.dp_boolean), ('_bake_ok', InternalTraversal.dp_boolean), ('_current_path', InternalTraversal.dp_has_cache_key), ('_enable_single_crit', InternalTraversal.dp_boolean), ('_enable_eagerloads', InternalTraversal.dp_boolean), ('_only_load_props', InternalTraversal.dp_plain_obj), ('_set_base_alias', InternalTraversal.dp_boolean), ('_for_refresh_state', InternalTraversal.dp_boolean), ('_render_for_subquery', InternalTraversal.dp_boolean), ('_is_star', InternalTraversal.dp_boolean)]
    _use_legacy_query_style = False
    _for_statement = False
    _bake_ok = True
    _current_path = _path_registry
    _enable_single_crit = True
    _enable_eagerloads = True
    _only_load_props = None
    _set_base_alias = False
    _for_refresh_state = False
    _render_for_subquery = False
    _is_star = False