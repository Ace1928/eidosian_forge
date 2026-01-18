from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifacts_table():
    op.create_table('artifacts', Column('id', String(length=36), nullable=False), Column('name', String(length=255), nullable=False), Column('type_name', String(length=255), nullable=False), Column('type_version_prefix', BigInteger(), nullable=False), Column('type_version_suffix', String(length=255), nullable=True), Column('type_version_meta', String(length=255), nullable=True), Column('version_prefix', BigInteger(), nullable=False), Column('version_suffix', String(length=255), nullable=True), Column('version_meta', String(length=255), nullable=True), Column('description', Text(), nullable=True), Column('visibility', String(length=32), nullable=False), Column('state', String(length=32), nullable=False), Column('owner', String(length=255), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), Column('deleted_at', DateTime(), nullable=True), Column('published_at', DateTime(), nullable=True), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_name_and_version', 'artifacts', ['name', 'version_prefix', 'version_suffix'], unique=False)
    op.create_index('ix_artifact_owner', 'artifacts', ['owner'], unique=False)
    op.create_index('ix_artifact_state', 'artifacts', ['state'], unique=False)
    op.create_index('ix_artifact_type', 'artifacts', ['type_name', 'type_version_prefix', 'type_version_suffix'], unique=False)
    op.create_index('ix_artifact_visibility', 'artifacts', ['visibility'], unique=False)