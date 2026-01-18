from alembic import op
from sqlalchemy import Column, Enum
from glance.cmd import manage
from glance.db import migration
from glance.db.sqlalchemy.schema import Boolean
def _change_nullability_and_default_on_is_public():
    with op.batch_alter_table('images') as batch_op:
        batch_op.alter_column('is_public', nullable=True, server_default=None, existing_type=Boolean())