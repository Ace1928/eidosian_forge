from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
def _on_conflict_target(self, clause, **kw):
    if clause.constraint_target is not None:
        target_text = '(%s)' % clause.constraint_target
    elif clause.inferred_target_elements is not None:
        target_text = '(%s)' % ', '.join((self.preparer.quote(c) if isinstance(c, str) else self.process(c, include_table=False, use_schema=False) for c in clause.inferred_target_elements))
        if clause.inferred_target_whereclause is not None:
            target_text += ' WHERE %s' % self.process(clause.inferred_target_whereclause, include_table=False, use_schema=False, literal_binds=True)
    else:
        target_text = ''
    return target_text