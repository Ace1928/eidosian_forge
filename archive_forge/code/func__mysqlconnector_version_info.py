import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
@util.memoized_property
def _mysqlconnector_version_info(self):
    if self.dbapi and hasattr(self.dbapi, '__version__'):
        m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', self.dbapi.__version__)
        if m:
            return tuple((int(x) for x in m.group(1, 2, 3) if x is not None))