import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def _get_dns_domain_config():
    if not cfg.CONF.dns_domain:
        return ''
    if cfg.CONF.dns_domain.endswith('.'):
        return cfg.CONF.dns_domain
    return '%s.' % cfg.CONF.dns_domain