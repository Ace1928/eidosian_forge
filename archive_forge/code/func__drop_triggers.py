from alembic import op
from sqlalchemy import Enum
from glance.cmd import manage
from glance.db import migration
def _drop_triggers(connection):
    engine_name = connection.engine.name
    if engine_name == 'mysql':
        op.execute(MYSQL_DROP_INSERT_TRIGGER)
        op.execute(MYSQL_DROP_UPDATE_TRIGGER)