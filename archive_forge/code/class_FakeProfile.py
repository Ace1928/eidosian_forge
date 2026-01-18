from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import profile as sp
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeProfile(object):

    def __init__(self, id='some_id', spec=None):
        self.id = id
        self.name = 'SenlinProfile'
        self.metadata = {}
        self.spec = spec or profile_spec