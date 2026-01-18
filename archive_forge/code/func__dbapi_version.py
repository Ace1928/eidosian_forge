import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
@util.memoized_property
def _dbapi_version(self):
    if self.dbapi and hasattr(self.dbapi, '__version__'):
        return tuple([int(x) for x in re.findall('(\\d+)(?:[-\\.]?|$)', self.dbapi.__version__)])
    else:
        return (99, 99, 99)