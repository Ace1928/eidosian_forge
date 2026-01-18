from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import resource
from heat.engine.resources.openstack.mistral import external_resource
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class FakeExecution(object):

    def __init__(self, id='1234', output='{}', state='IDLE'):
        self.id = id
        self.output = output
        self.state = state