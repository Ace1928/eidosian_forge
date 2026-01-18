from keystone.common import sql
from keystone import exception
from keystone.policy.backends import rules
class PolicyModel(sql.ModelBase, sql.ModelDictMixinWithExtras):
    __tablename__ = 'policy'
    attributes = ['id', 'blob', 'type']
    id = sql.Column(sql.String(64), primary_key=True)
    blob = sql.Column(sql.JsonBlob(), nullable=False)
    type = sql.Column(sql.String(255), nullable=False)
    extra = sql.Column(sql.JsonBlob())