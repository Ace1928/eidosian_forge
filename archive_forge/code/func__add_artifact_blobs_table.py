from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifact_blobs_table():
    op.create_table('artifact_blobs', Column('id', String(length=36), nullable=False), Column('artifact_id', String(length=36), nullable=False), Column('size', BigInteger(), nullable=False), Column('checksum', String(length=32), nullable=True), Column('name', String(length=255), nullable=False), Column('item_key', String(length=329), nullable=True), Column('position', Integer(), nullable=True), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), ForeignKeyConstraint(['artifact_id'], ['artifacts.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_blobs_artifact_id', 'artifact_blobs', ['artifact_id'], unique=False)
    op.create_index('ix_artifact_blobs_name', 'artifact_blobs', ['name'], unique=False)