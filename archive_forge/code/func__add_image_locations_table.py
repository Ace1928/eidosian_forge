from alembic import op
from sqlalchemy import sql
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_image_locations_table():
    op.create_table('image_locations', Column('id', Integer(), nullable=False), Column('image_id', String(length=36), nullable=False), Column('value', Text(), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), Column('deleted_at', DateTime(), nullable=True), Column('deleted', Boolean(), nullable=False), Column('meta_data', JSONEncodedDict(), nullable=True), Column('status', String(length=30), server_default='active', nullable=False), PrimaryKeyConstraint('id'), ForeignKeyConstraint(['image_id'], ['images.id']), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_image_locations_deleted', 'image_locations', ['deleted'], unique=False)
    op.create_index('ix_image_locations_image_id', 'image_locations', ['image_id'], unique=False)