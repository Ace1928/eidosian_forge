from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_tasks_table():
    op.create_table('tasks', Column('id', String(length=36), nullable=False), Column('type', String(length=30), nullable=False), Column('status', String(length=30), nullable=False), Column('owner', String(length=255), nullable=False), Column('expires_at', DateTime(), nullable=True), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), Column('deleted_at', DateTime(), nullable=True), Column('deleted', Boolean(), nullable=False), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_tasks_deleted', 'tasks', ['deleted'], unique=False)
    op.create_index('ix_tasks_owner', 'tasks', ['owner'], unique=False)
    op.create_index('ix_tasks_status', 'tasks', ['status'], unique=False)
    op.create_index('ix_tasks_type', 'tasks', ['type'], unique=False)
    op.create_index('ix_tasks_updated_at', 'tasks', ['updated_at'], unique=False)