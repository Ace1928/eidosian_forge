from alembic import op
from sqlalchemy import Column, Enum
from glance.cmd import manage
from glance.db import migration
from glance.db.sqlalchemy.schema import Boolean
def _add_triggers(connection):
    if connection.engine.name == 'mysql':
        op.execute(MYSQL_INSERT_TRIGGER % (ERROR_MESSAGE, ERROR_MESSAGE, ERROR_MESSAGE))
        op.execute(MYSQL_UPDATE_TRIGGER % (ERROR_MESSAGE, ERROR_MESSAGE, ERROR_MESSAGE, ERROR_MESSAGE))