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
class QueryContext:
    __slots__ = ('top_level_context', 'compile_state', 'query', 'params', 'load_options', 'bind_arguments', 'execution_options', 'session', 'autoflush', 'populate_existing', 'invoke_all_eagers', 'version_check', 'refresh_state', 'create_eager_joins', 'propagated_loader_options', 'attributes', 'runid', 'partials', 'post_load_paths', 'identity_token', 'yield_per', 'loaders_require_buffering', 'loaders_require_uniquing')
    runid: int
    post_load_paths: Dict[PathRegistry, PostLoad]
    compile_state: ORMCompileState

    class default_load_options(Options):
        _only_return_tuples = False
        _populate_existing = False
        _version_check = False
        _invoke_all_eagers = True
        _autoflush = True
        _identity_token = None
        _yield_per = None
        _refresh_state = None
        _lazy_loaded_from = None
        _legacy_uniquing = False
        _sa_top_level_orm_context = None
        _is_user_refresh = False

    def __init__(self, compile_state: CompileState, statement: Union[Select[Any], FromStatement[Any]], params: _CoreSingleExecuteParams, session: Session, load_options: Union[Type[QueryContext.default_load_options], QueryContext.default_load_options], execution_options: Optional[OrmExecuteOptionsParameter]=None, bind_arguments: Optional[_BindArguments]=None):
        self.load_options = load_options
        self.execution_options = execution_options or _EMPTY_DICT
        self.bind_arguments = bind_arguments or _EMPTY_DICT
        self.compile_state = compile_state
        self.query = statement
        self.session = session
        self.loaders_require_buffering = False
        self.loaders_require_uniquing = False
        self.params = params
        self.top_level_context = load_options._sa_top_level_orm_context
        cached_options = compile_state.select_statement._with_options
        uncached_options = statement._with_options
        self.propagated_loader_options = tuple((opt._adapt_cached_option_to_uncached_option(self, uncached_opt) for opt, uncached_opt in zip(cached_options, uncached_options) if opt.propagate_to_loaders))
        self.attributes = dict(compile_state.attributes)
        self.autoflush = load_options._autoflush
        self.populate_existing = load_options._populate_existing
        self.invoke_all_eagers = load_options._invoke_all_eagers
        self.version_check = load_options._version_check
        self.refresh_state = load_options._refresh_state
        self.yield_per = load_options._yield_per
        self.identity_token = load_options._identity_token

    def _get_top_level_context(self) -> QueryContext:
        return self.top_level_context or self