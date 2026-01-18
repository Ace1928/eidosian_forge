from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.resource.config_backends import base
def _delete_config_options(self, session, domain_id, group, option):
    for config_table in [WhiteListedConfig, SensitiveConfig]:
        query = session.query(config_table)
        query = query.filter_by(domain_id=domain_id)
        if group:
            query = query.filter_by(group=group)
            if option:
                query = query.filter_by(option=option)
        query.delete(False)