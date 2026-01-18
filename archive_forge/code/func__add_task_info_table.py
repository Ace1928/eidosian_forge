from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_task_info_table():
    op.create_table('task_info', Column('task_id', String(length=36), nullable=False), Column('input', JSONEncodedDict(), nullable=True), Column('result', JSONEncodedDict(), nullable=True), Column('message', Text(), nullable=True), ForeignKeyConstraint(['task_id'], ['tasks.id']), PrimaryKeyConstraint('task_id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)