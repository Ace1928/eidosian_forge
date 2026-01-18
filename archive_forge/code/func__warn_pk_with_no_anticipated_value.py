from __future__ import annotations
import functools
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import coercions
from . import dml
from . import elements
from . import roles
from .base import _DefaultDescriptionTuple
from .dml import isinsert as _compile_state_isinsert
from .elements import ColumnClause
from .schema import default_is_clause_element
from .schema import default_is_sequence
from .selectable import Select
from .selectable import TableClause
from .. import exc
from .. import util
from ..util.typing import Literal
def _warn_pk_with_no_anticipated_value(c):
    msg = "Column '%s.%s' is marked as a member of the primary key for table '%s', but has no Python-side or server-side default generator indicated, nor does it indicate 'autoincrement=True' or 'nullable=True', and no explicit value is passed.  Primary key columns typically may not store NULL." % (c.table.fullname, c.name, c.table.fullname)
    if len(c.table.primary_key) > 1:
        msg += " Note that as of SQLAlchemy 1.1, 'autoincrement=True' must be indicated explicitly for composite (e.g. multicolumn) primary keys if AUTO_INCREMENT/SERIAL/IDENTITY behavior is expected for one of the columns in the primary key. CREATE TABLE statements are impacted by this change as well on most backends."
    util.warn(msg)