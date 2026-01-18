from unittest import mock
from openstack import exceptions
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import receiver as sr
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_recv(self, template):
    self.stack = utils.parse_stack(template)
    recv = self.stack['senlin-receiver']
    return recv