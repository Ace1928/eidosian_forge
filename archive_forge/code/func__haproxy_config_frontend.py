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
def _haproxy_config_frontend(self):
    listener = self.properties[self.LISTENERS][0]
    lb_port = listener[self.LISTENER_LOAD_BALANCER_PORT]
    return '\nfrontend http\n    bind *:%s\n    default_backend servers\n' % lb_port