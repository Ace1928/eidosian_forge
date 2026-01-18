import functools
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class DummyThreadGroupMgrLogStart(DummyThreadGroupManager):

    def __init__(self):
        super(DummyThreadGroupMgrLogStart, self).__init__()
        self.started = []

    def start_with_lock(self, cnxt, stack, engine_id, func, *args, **kwargs):
        self.started.append((stack.id, func))
        return DummyThread()

    def start_with_acquired_lock(self, stack, lock, func, *args, **kwargs):
        self.started.append((stack.id, func))
        return DummyThread()

    def start(self, stack_id, func, *args, **kwargs):
        self.started.append((stack_id, func))