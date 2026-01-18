import configparser
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.i18n import _, _LE
def _form_default_params(self):
    default = {}
    if CONF.swift_store_user and CONF.swift_store_key and CONF.swift_store_auth_address:
        default['user'] = CONF.swift_store_user
        default['key'] = CONF.swift_store_key
        default['auth_address'] = CONF.swift_store_auth_address
        return {CONF.default_swift_reference: default}
    return {}