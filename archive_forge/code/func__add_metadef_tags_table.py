from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_metadef_tags_table():
    op.create_table('metadef_tags', Column('id', Integer(), nullable=False), Column('namespace_id', Integer(), nullable=False), Column('name', String(length=80), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), ForeignKeyConstraint(['namespace_id'], ['metadef_namespaces.id']), PrimaryKeyConstraint('id'), UniqueConstraint('namespace_id', 'name', name='uq_metadef_tags_namespace_id_name'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_metadef_tags_name', 'metadef_tags', ['name'], unique=False)