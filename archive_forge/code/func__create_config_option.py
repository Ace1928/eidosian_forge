from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.resource.config_backends import base
def _create_config_option(self, session, domain_id, group, option, sensitive, value):
    config_table = self.choose_table(sensitive)
    ref = config_table(domain_id=domain_id, group=group, option=option, value=value)
    session.add(ref)