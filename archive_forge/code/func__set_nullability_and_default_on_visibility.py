from alembic import op
from sqlalchemy import Enum
from glance.cmd import manage
from glance.db import migration
def _set_nullability_and_default_on_visibility():
    existing_type = Enum('private', 'public', 'shared', 'community', name='image_visibility')
    with op.batch_alter_table('images') as batch_op:
        batch_op.alter_column('visibility', nullable=False, server_default='shared', existing_type=existing_type)