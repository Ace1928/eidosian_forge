from keystone.common import sql
from keystone.identity.mapping_backends import base
from keystone.identity.mapping_backends import mapping as identity_mapping
class IDMapping(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'id_mapping'
    public_id = sql.Column(sql.String(64), primary_key=True)
    domain_id = sql.Column(sql.String(64), nullable=False)
    local_id = sql.Column(sql.String(255), nullable=False)
    entity_type = sql.Column(sql.Enum(identity_mapping.EntityType.USER, identity_mapping.EntityType.GROUP, name='entity_type'), nullable=False)
    __table_args__ = (sql.UniqueConstraint('domain_id', 'local_id', 'entity_type'),)