import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class LoadbalancerReloadFallbackTest(LoadbalancerReloadTest):

    def setup_mocks(self, group, member_refids):
        group.get_output = mock.Mock(side_effect=exception.NotFound)

        def make_mock_member(refid):
            mem = mock.Mock()
            mem.FnGetRefId = mock.Mock(return_value=refid)
            return mem
        members = [make_mock_member(r) for r in member_refids]
        mock_members = self.patchobject(grouputils, 'get_members', return_value=members)
        return mock_members

    def check_mocks(self, group, mock_members):
        mock_members.assert_called_once_with(group)