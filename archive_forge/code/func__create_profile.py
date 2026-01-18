from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import profile as sp
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_profile(self, template):
    profile = self._init_profile(template)
    self.senlin_mock.create_profile.return_value = self.fake_p
    scheduler.TaskRunner(profile.create)()
    self.assertEqual((profile.CREATE, profile.COMPLETE), profile.state)
    self.assertEqual(self.fake_p.id, profile.resource_id)
    return profile