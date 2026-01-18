import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupNameListTest(common.HeatTestCase):
    """This class tests ResourceGroup._resource_names()."""
    scenarios = [('1', dict(skiplist=[], count=0, expected=[])), ('2', dict(skiplist=[], count=4, expected=['0', '1', '2', '3'])), ('3', dict(skiplist=['5', '6'], count=3, expected=['0', '1', '2'])), ('4', dict(skiplist=['2', '4'], count=4, expected=['0', '1', '3', '5']))]

    def test_names(self):
        stack = utils.parse_stack(template)
        resg = stack['group1']
        resg.properties = mock.MagicMock()
        resg.properties.get.return_value = self.count
        resg._name_skiplist = mock.MagicMock(return_value=self.skiplist)
        self.assertEqual(self.expected, list(resg._resource_names()))