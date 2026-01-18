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
class LoadbalancerReloadTest(common.HeatTestCase):

    def test_Instances(self):
        t = template_format.parse(inline_templates.as_template)
        stack = utils.parse_stack(t)
        lb = stack['ElasticLoadBalancer']
        lb.update = mock.Mock(return_value=None)
        defn = rsrc_defn.ResourceDefinition('asg', 'OS::Heat::InstanceGroup', {'Size': 2, 'AvailabilityZones': ['zoneb'], 'LaunchConfigurationName': 'LaunchConfig', 'LoadBalancerNames': ['ElasticLoadBalancer']})
        group = instgrp.InstanceGroup('asg', defn, stack)
        mocks = self.setup_mocks(group, ['aaaa', 'bbb'])
        expected = rsrc_defn.ResourceDefinition('ElasticLoadBalancer', 'AWS::ElasticLoadBalancing::LoadBalancer', {'Instances': ['aaaa', 'bbb'], 'Listeners': [{'InstancePort': u'80', 'LoadBalancerPort': u'80', 'Protocol': 'HTTP'}], 'AvailabilityZones': ['nova']})
        group._lb_reload()
        self.check_mocks(group, mocks)
        lb.update.assert_called_once_with(expected)

    def test_lb_reload_static_resolve(self):
        t = template_format.parse(inline_templates.as_template)
        properties = t['Resources']['ElasticLoadBalancer']['Properties']
        properties['AvailabilityZones'] = {'Fn::GetAZs': ''}
        self.patchobject(stk_defn.StackDefinition, 'get_availability_zones', return_value=['abc', 'xyz'])
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        lb = stack['ElasticLoadBalancer']
        lb.state_set(lb.CREATE, lb.COMPLETE)
        lb.handle_update = mock.Mock(return_value=None)
        group = stack['WebServerGroup']
        self.setup_mocks(group, ['aaaabbbbcccc'])
        group._lb_reload()
        lb.handle_update.assert_called_once_with(mock.ANY, mock.ANY, {'Instances': ['aaaabbbbcccc']})

    def setup_mocks(self, group, member_refids):
        refs = {str(i): r for i, r in enumerate(member_refids)}
        group.get_output = mock.Mock(return_value=refs)
        names = sorted(refs.keys())
        group_data = group._group_data()
        group_data.member_names = mock.Mock(return_value=names)
        group._group_data = mock.Mock(return_value=group_data)

    def check_mocks(self, group, unused):
        pass