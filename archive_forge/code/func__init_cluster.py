from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import cluster as sc
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_cluster(self, template, name='super-cluster'):
    self.stack = utils.parse_stack(template)
    cluster = self.stack[name]
    return cluster