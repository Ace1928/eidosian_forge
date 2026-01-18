import re
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .base import MySQLIdentifierPreparer
from .base import TEXT
from ... import sql
from ... import util
def _parse_dbapi_version(self, version):
    m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', version)
    if m:
        return tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
    else:
        return (0, 0, 0)