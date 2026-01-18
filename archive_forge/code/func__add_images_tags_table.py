from alembic import op
from sqlalchemy import sql
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_images_tags_table():
    op.create_table('image_tags', Column('id', Integer(), nullable=False), Column('image_id', String(length=36), nullable=False), Column('value', String(length=255), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), Column('deleted_at', DateTime(), nullable=True), Column('deleted', Boolean(), nullable=False), ForeignKeyConstraint(['image_id'], ['images.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_image_tags_image_id', 'image_tags', ['image_id'], unique=False)
    op.create_index('ix_image_tags_image_id_tag_value', 'image_tags', ['image_id', 'value'], unique=False)