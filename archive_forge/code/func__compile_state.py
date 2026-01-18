from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self
def _compile_state(self, for_statement: bool=False, **kw: Any) -> ORMCompileState:
    """Create an out-of-compiler ORMCompileState object.

        The ORMCompileState object is normally created directly as a result
        of the SQLCompiler.process() method being handed a Select()
        or FromStatement() object that uses the "orm" plugin.   This method
        provides a means of creating this ORMCompileState object directly
        without using the compiler.

        This method is used only for deprecated cases, which include
        the .from_self() method for a Query that has multiple levels
        of .from_self() in use, as well as the instances() method.  It is
        also used within the test suite to generate ORMCompileState objects
        for test purposes.

        """
    stmt = self._statement_20(for_statement=for_statement, **kw)
    assert for_statement == stmt._compile_options._for_statement
    compile_state_cls = cast(ORMCompileState, ORMCompileState._get_plugin_class_for_plugin(stmt, 'orm'))
    return compile_state_cls.create_for_statement(stmt, None)