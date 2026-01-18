from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
class TestDescribeLoadBalancers(unittest.TestCase):

    def test_other_policy(self):
        elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        mock_response = mock.Mock()
        mock_response.read.return_value = DESCRIBE_RESPONSE
        mock_response.status = 200
        elb.make_request = mock.Mock(return_value=mock_response)
        load_balancers = elb.get_all_load_balancers()
        self.assertEqual(len(load_balancers), 1)
        lb = load_balancers[0]
        self.assertEqual(len(lb.policies.other_policies), 2)
        self.assertEqual(lb.policies.other_policies[0].policy_name, 'AWSConsole-SSLNegotiationPolicy-my-test-loadbalancer')
        self.assertEqual(lb.policies.other_policies[1].policy_name, 'EnableProxyProtocol')
        self.assertEqual(len(lb.backends), 1)
        self.assertEqual(len(lb.backends[0].policies), 1)
        self.assertEqual(lb.backends[0].policies[0].policy_name, 'EnableProxyProtocol')
        self.assertEqual(lb.backends[0].instance_port, 80)

    def test_request_with_marker(self):
        elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        mock_response = mock.Mock()
        mock_response.read.return_value = DESCRIBE_RESPONSE
        mock_response.status = 200
        elb.make_request = mock.Mock(return_value=mock_response)
        load_balancers1 = elb.get_all_load_balancers()
        self.assertEqual('1234', load_balancers1.marker)
        load_balancers2 = elb.get_all_load_balancers(marker=load_balancers1.marker)
        self.assertEqual(len(load_balancers2), 1)