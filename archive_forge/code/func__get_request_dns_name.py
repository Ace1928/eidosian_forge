import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def _get_request_dns_name(dns_name):
    dns_domain = _get_dns_domain_config()
    if dns_domain and dns_domain != constants.DNS_DOMAIN_DEFAULT:
        return dns_name
    return ''