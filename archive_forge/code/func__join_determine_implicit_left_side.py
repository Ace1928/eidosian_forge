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
def _join_determine_implicit_left_side(self, entities_collection, left, right, onclause):
    """When join conditions don't express the left side explicitly,
        determine if an existing FROM or entity in this query
        can serve as the left hand side.

        """
    r_info = inspect(right)
    replace_from_obj_index = use_entity_index = None
    if self.from_clauses:
        indexes = sql_util.find_left_clause_to_join_from(self.from_clauses, r_info.selectable, onclause)
        if len(indexes) == 1:
            replace_from_obj_index = indexes[0]
            left = self.from_clauses[replace_from_obj_index]
        elif len(indexes) > 1:
            raise sa_exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
        else:
            raise sa_exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
    elif entities_collection:
        potential = {}
        for entity_index, ent in enumerate(entities_collection):
            entity = ent.entity_zero_or_selectable
            if entity is None:
                continue
            ent_info = inspect(entity)
            if ent_info is r_info:
                continue
            if isinstance(ent, _MapperEntity):
                potential[ent.selectable] = (entity_index, entity)
            else:
                potential[ent_info.selectable] = (None, entity)
        all_clauses = list(potential.keys())
        indexes = sql_util.find_left_clause_to_join_from(all_clauses, r_info.selectable, onclause)
        if len(indexes) == 1:
            use_entity_index, left = potential[all_clauses[indexes[0]]]
        elif len(indexes) > 1:
            raise sa_exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
        else:
            raise sa_exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
    else:
        raise sa_exc.InvalidRequestError('No entities to join from; please use select_from() to establish the left entity/selectable of this join')
    return (left, replace_from_obj_index, use_entity_index)