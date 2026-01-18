import os
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import stack_resource
def _haproxy_config_backend(self):
    health_chk = self.properties[self.HEALTH_CHECK]
    if health_chk:
        timeout = int(health_chk[self.HEALTH_CHECK_TIMEOUT])
        timeout_check = 'timeout check %ds' % timeout
        spaces = '    '
    else:
        timeout_check = ''
        spaces = ''
    return '\nbackend servers\n    balance roundrobin\n    option http-server-close\n    option forwardfor\n    option httpchk\n%s%s\n' % (spaces, timeout_check)