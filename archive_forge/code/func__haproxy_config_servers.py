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
def _haproxy_config_servers(self, instances):
    listener = self.properties[self.LISTENERS][0]
    inst_port = listener[self.LISTENER_INSTANCE_PORT]
    spaces = '    '
    check = ''
    health_chk = self.properties[self.HEALTH_CHECK]
    if health_chk:
        check = ' check inter %ss fall %s rise %s' % (health_chk[self.HEALTH_CHECK_INTERVAL], health_chk[self.HEALTH_CHECK_UNHEALTHY_THRESHOLD], health_chk[self.HEALTH_CHECK_HEALTHY_THRESHOLD])
    servers = []
    n = 1
    nova_cp = self.client_plugin('nova')
    for i in instances or []:
        ip = nova_cp.server_to_ipaddress(i) or '0.0.0.0'
        LOG.debug('haproxy server:%s', ip)
        servers.append('%sserver server%d %s:%s%s' % (spaces, n, ip, inst_port, check))
        n = n + 1
    return '\n'.join(servers)