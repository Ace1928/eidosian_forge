from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class TestIPCIDRConstraint(common.HeatTestCase):

    def setUp(self):
        super(TestIPCIDRConstraint, self).setUp()
        self.constraint = cc.IPCIDRConstraint()

    def test_valid_format(self):
        validate_format = ['10.0.0.0/24', '1.1.1.1', '1.0.1.1', '255.255.255.255', '6000::/64', '2002:2002::20c:29ff:fe7d:811a', '::1', '2002::', '2002::1']
        for value in validate_format:
            self.assertTrue(self.constraint.validate(value, None))

    def test_invalid_format(self):
        invalidate_format = ['::/129', 'Invalid cidr', '300.0.0.0/24', '10.0.0.0/33', '10.0.0/24', '10.0/24', '10.0.a.10/24', '8.8.8.0/ 24', None, 123, '1.1', '1.1.', '1.1.1', '1.1.1.', '1.1.1.256', '1.a.1.1', '2002::2001::1', '2002::g', '2001::0::', '20c:29ff:fe7d:811a']
        for value in invalidate_format:
            self.assertFalse(self.constraint.validate(value, None))

    @mock.patch('neutron_lib.api.validators.validate_subnet')
    @mock.patch('neutron_lib.api.validators.validate_ip_address')
    def test_validate(self, mock_validate_ip, mock_validate_subnet):
        test_formats = ['10.0.0/24', '10.0/24', '10.0.0.0/33']
        for cidr in test_formats:
            self.assertFalse(self.constraint.validate(cidr, None))
            mock_validate_subnet.assert_called_with(cidr)
        self.assertEqual(mock_validate_subnet.call_count, 3)
        test_formats = ['10.0.0', '10.0', '10.0.0.0']
        for ip in test_formats:
            self.assertFalse(self.constraint.validate(ip, None))
            mock_validate_ip.assert_called_with(ip)
        self.assertEqual(mock_validate_ip.call_count, 3)