from alembic import op
from sqlalchemy.schema import (
from glance.db.sqlalchemy.schema import (
from glance.db.sqlalchemy.models import JSONEncodedDict
def _add_metadef_namespaces_table():
    op.create_table('metadef_namespaces', Column('id', Integer(), nullable=False), Column('namespace', String(length=80), nullable=False), Column('display_name', String(length=80), nullable=True), Column('description', Text(), nullable=True), Column('visibility', String(length=32), nullable=True), Column('protected', Boolean(), nullable=True), Column('owner', String(length=255), nullable=False), Column('created_at', DateTime(), nullable=False), Column('updated_at', DateTime(), nullable=True), PrimaryKeyConstraint('id'), UniqueConstraint('namespace', name='uq_metadef_namespaces_namespace'), mysql_engine='InnoDB', mysql_charset='utf8', extend_existing=True)
    op.create_index('ix_metadef_namespaces_owner', 'metadef_namespaces', ['owner'], unique=False)