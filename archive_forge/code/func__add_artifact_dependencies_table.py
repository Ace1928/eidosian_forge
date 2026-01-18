from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifact_dependencies_table():
    op.create_table('artifact_dependencies', Column('id', String(length=36), nullable=False), Column('artifact_source', String(length=36), nullable=False), Column('artifact_dest', String(length=36), nullable=False), Column('artifact_origin', String(length=36), nullable=False), Column('is_direct', Boolean(), nullable=False), Column('position', Integer(), nullable=True), Column('name', String(length=36), nullable=True), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), ForeignKeyConstraint(['artifact_dest'], ['artifacts.id']), ForeignKeyConstraint(['artifact_origin'], ['artifacts.id']), ForeignKeyConstraint(['artifact_source'], ['artifacts.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_dependencies_dest_id', 'artifact_dependencies', ['artifact_dest'], unique=False)
    op.create_index('ix_artifact_dependencies_direct_dependencies', 'artifact_dependencies', ['artifact_source', 'is_direct'], unique=False)
    op.create_index('ix_artifact_dependencies_origin_id', 'artifact_dependencies', ['artifact_origin'], unique=False)
    op.create_index('ix_artifact_dependencies_source_id', 'artifact_dependencies', ['artifact_source'], unique=False)