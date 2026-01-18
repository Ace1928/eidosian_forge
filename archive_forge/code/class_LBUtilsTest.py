from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import properties
from heat.engine import resource
from heat.scaling import lbutils
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class LBUtilsTest(common.HeatTestCase):

    def setUp(self):
        super(LBUtilsTest, self).setUp()
        resource._register_class('Mock::AWS::LB', MockedAWSLB)
        resource._register_class('Mock::Not::LB', generic_resource.GenericResource)
        t = template_format.parse(lb_stack)
        self.stack = utils.parse_stack(t)

    def test_reload_aws_lb(self):
        id_list = ['ID1', 'ID2', 'ID3']
        lb1 = self.stack['aws_lb_1']
        lb2 = self.stack['aws_lb_2']
        lb1.action = mock.Mock(return_value=lb1.CREATE)
        lb2.action = mock.Mock(return_value=lb2.CREATE)
        lb1.handle_update = mock.Mock()
        lb2.handle_update = mock.Mock()
        prop_diff = {'Instances': id_list}
        lbutils.reconfigure_loadbalancers([lb1, lb2], id_list)
        lb1.handle_update.assert_called_with(mock.ANY, mock.ANY, prop_diff)
        lb2.handle_update.assert_called_with(mock.ANY, mock.ANY, prop_diff)

    def test_reload_non_lb(self):
        id_list = ['ID1', 'ID2', 'ID3']
        non_lb = self.stack['non_lb']
        error = self.assertRaises(exception.Error, lbutils.reconfigure_loadbalancers, [non_lb], id_list)
        self.assertIn("Unsupported resource 'non_lb' in LoadBalancerNames", str(error))