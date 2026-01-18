from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeNodeGroupTemplate(object):

    def __init__(self):
        self.id = 'some_ng_id'
        self.name = 'test-cluster-template'
        self.to_dict = lambda: {'ng-template': 'info'}