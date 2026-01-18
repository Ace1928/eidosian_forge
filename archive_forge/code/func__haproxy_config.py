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
def _haproxy_config(self, instances):
    return '%s%s%s%s\n' % (self._haproxy_config_global(), self._haproxy_config_frontend(), self._haproxy_config_backend(), self._haproxy_config_servers(instances))