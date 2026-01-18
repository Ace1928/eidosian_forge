from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class TestIPConstraint(common.HeatTestCase):

    def setUp(self):
        super(TestIPConstraint, self).setUp()
        self.constraint = cc.IPConstraint()

    def test_validate_ipv4_format(self):
        validate_format = ['1.1.1.1', '1.0.1.1', '255.255.255.255']
        for ip in validate_format:
            self.assertTrue(self.constraint.validate(ip, None))

    def test_invalidate_ipv4_format(self):
        invalidate_format = [None, 123, '1.1', '1.1.', '1.1.1', '1.1.1.', '1.1.1.256', 'invalidate format', '1.a.1.1']
        for ip in invalidate_format:
            self.assertFalse(self.constraint.validate(ip, None))

    def test_validate_ipv6_format(self):
        validate_format = ['2002:2002::20c:29ff:fe7d:811a', '::1', '2002::', '2002::1']
        for ip in validate_format:
            self.assertTrue(self.constraint.validate(ip, None))

    def test_invalidate_ipv6_format(self):
        invalidate_format = ['2002::2001::1', '2002::g', 'invalidate format', '2001::0::', '20c:29ff:fe7d:811a']
        for ip in invalidate_format:
            self.assertFalse(self.constraint.validate(ip, None))