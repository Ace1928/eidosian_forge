import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
def _is_server_side(self, cursor):
    if self.engine.dialect.driver == 'psycopg2':
        return bool(cursor.name)
    elif self.engine.dialect.driver == 'pymysql':
        sscursor = __import__('pymysql.cursors').cursors.SSCursor
        return isinstance(cursor, sscursor)
    elif self.engine.dialect.driver in ('aiomysql', 'asyncmy', 'aioodbc'):
        return cursor.server_side
    elif self.engine.dialect.driver == 'mysqldb':
        sscursor = __import__('MySQLdb.cursors').cursors.SSCursor
        return isinstance(cursor, sscursor)
    elif self.engine.dialect.driver == 'mariadbconnector':
        return not cursor.buffered
    elif self.engine.dialect.driver in ('asyncpg', 'aiosqlite'):
        return cursor.server_side
    elif self.engine.dialect.driver == 'pg8000':
        return getattr(cursor, 'server_side', False)
    elif self.engine.dialect.driver == 'psycopg':
        return bool(getattr(cursor, 'name', False))
    else:
        return False