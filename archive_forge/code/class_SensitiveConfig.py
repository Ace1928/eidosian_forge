from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.resource.config_backends import base
class SensitiveConfig(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'sensitive_config'
    domain_id = sql.Column(sql.String(64), primary_key=True)
    group = sql.Column(sql.String(255), primary_key=True)
    option = sql.Column(sql.String(255), primary_key=True)
    value = sql.Column(sql.JsonBlob(), nullable=False)

    def to_dict(self):
        d = super(SensitiveConfig, self).to_dict()
        d.pop('domain_id')
        return d