import copy
from unittest import mock
from openstack.clustering.v1._proxy import Proxy
from openstack import exceptions
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import policy
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class FakePolicy(object):

    def __init__(self, id='some_id', spec=None):
        self.id = id
        self.name = 'SenlinPolicy'

    def to_dict(self):
        return {'id': self.id, 'name': self.name}