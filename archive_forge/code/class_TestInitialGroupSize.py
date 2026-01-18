import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestInitialGroupSize(common.HeatTestCase):
    scenarios = [('000', dict(mins=0, maxs=0, desired=0, expected=0)), ('040', dict(mins=0, maxs=4, desired=0, expected=0)), ('253', dict(mins=2, maxs=5, desired=3, expected=3)), ('14n', dict(mins=1, maxs=4, desired=None, expected=1))]

    def test_initial_size(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['MinSize'] = self.mins
        properties['MaxSize'] = self.maxs
        properties['DesiredCapacity'] = self.desired
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        group = stack['WebServerGroup']
        with mock.patch.object(group, '_create_template') as mock_cre_temp:
            group.child_template()
            mock_cre_temp.assert_called_once_with(self.expected)