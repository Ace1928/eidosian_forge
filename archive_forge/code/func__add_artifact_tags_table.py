from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
def _add_artifact_tags_table():
    op.create_table('artifact_tags', Column('id', String(length=36), nullable=False), Column('artifact_id', String(length=36), nullable=False), Column('value', String(length=255), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=False), ForeignKeyConstraint(['artifact_id'], ['artifacts.id']), PrimaryKeyConstraint('id'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_artifact_tags_artifact_id', 'artifact_tags', ['artifact_id'], unique=False)
    op.create_index('ix_artifact_tags_artifact_id_tag_value', 'artifact_tags', ['artifact_id', 'value'], unique=False)