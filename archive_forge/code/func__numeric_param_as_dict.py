from the connection pool, such as when using an ORM :class:`.Session` where
from working correctly.  The pysqlite DBAPI driver has several
import math
import os
import re
from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util
def _numeric_param_as_dict(parameters):
    if parameters:
        assert isinstance(parameters, tuple)
        return {str(idx): value for idx, value in enumerate(parameters, 1)}
    else:
        return ()