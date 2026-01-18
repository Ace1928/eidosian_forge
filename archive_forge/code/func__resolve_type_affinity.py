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
def _resolve_type_affinity(self, type_):
    """Return a data type from a reflected column, using affinity rules.

        SQLite's goal for universal compatibility introduces some complexity
        during reflection, as a column's defined type might not actually be a
        type that SQLite understands - or indeed, my not be defined *at all*.
        Internally, SQLite handles this with a 'data type affinity' for each
        column definition, mapping to one of 'TEXT', 'NUMERIC', 'INTEGER',
        'REAL', or 'NONE' (raw bits). The algorithm that determines this is
        listed in https://www.sqlite.org/datatype3.html section 2.1.

        This method allows SQLAlchemy to support that algorithm, while still
        providing access to smarter reflection utilities by recognizing
        column definitions that SQLite only supports through affinity (like
        DATE and DOUBLE).

        """
    match = re.match('([\\w ]+)(\\(.*?\\))?', type_)
    if match:
        coltype = match.group(1)
        args = match.group(2)
    else:
        coltype = ''
        args = ''
    if coltype in self.ischema_names:
        coltype = self.ischema_names[coltype]
    elif 'INT' in coltype:
        coltype = sqltypes.INTEGER
    elif 'CHAR' in coltype or 'CLOB' in coltype or 'TEXT' in coltype:
        coltype = sqltypes.TEXT
    elif 'BLOB' in coltype or not coltype:
        coltype = sqltypes.NullType
    elif 'REAL' in coltype or 'FLOA' in coltype or 'DOUB' in coltype:
        coltype = sqltypes.REAL
    else:
        coltype = sqltypes.NUMERIC
    if args is not None:
        args = re.findall('(\\d+)', args)
        try:
            coltype = coltype(*[int(a) for a in args])
        except TypeError:
            util.warn('Could not instantiate type %s with reflected arguments %s; using no arguments.' % (coltype, args))
            coltype = coltype()
    else:
        coltype = coltype()
    return coltype