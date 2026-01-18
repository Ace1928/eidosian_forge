from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifact_blob_locations_table():
    op.create_table('artifact_blob_locations', Column('id', String(length=36), nullable=False), Column('blob_id', String(length=36), nullable=False), Column('value', Text(), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), Column('position', Integer(), nullable=True), Column('status', String(length=36), nullable=True), ForeignKeyConstraint(['blob_id'], ['artifact_blobs.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_blob_locations_blob_id', 'artifact_blob_locations', ['blob_id'], unique=False)