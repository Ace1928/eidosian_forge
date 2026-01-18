import testtools
from unittest import mock
from troveclient.v1 import security_groups
class SecGroupRuleTest(testtools.TestCase):

    @mock.patch.object(security_groups.SecurityGroupRules, '__init__', mock.Mock(return_value=None))
    @mock.patch.object(security_groups.SecurityGroupRule, '__init__', mock.Mock(return_value=None))
    def setUp(self, *args):
        super(SecGroupRuleTest, self).setUp()
        self.security_group_rule = security_groups.SecurityGroupRule()
        self.security_group_rules = security_groups.SecurityGroupRules()

    def tearDown(self):
        super(SecGroupRuleTest, self).tearDown()

    def test___repr__(self):
        self.security_group_rule.group_id = 1
        self.security_group_rule.protocol = 'tcp'
        self.security_group_rule.from_port = 80
        self.security_group_rule.to_port = 80
        self.security_group_rule.cidr = '0.0.0.0//0'
        representation = '<SecurityGroupRule: ( Security Group id: %d, Protocol: %s, From_Port: %d, To_Port: %d,  CIDR: %s )>' % (1, 'tcp', 80, 80, '0.0.0.0//0')
        self.assertEqual(representation, self.security_group_rule.__repr__())

    def test_create(self):

        def side_effect_func(path, body, inst, return_raw=True):
            return (path, body, inst)
        self.security_group_rules._create = mock.Mock(side_effect=side_effect_func)
        p, b, i = self.security_group_rules.create(1, '0.0.0.0//0')
        self.assertEqual('/security-group-rules', p)
        self.assertEqual('security_group_rule', i)
        self.assertEqual(1, b['security_group_rule']['group_id'])
        self.assertEqual('0.0.0.0//0', b['security_group_rule']['cidr'])

    def test_delete(self):
        resp = mock.Mock()
        resp.status = 200
        body = None
        self.security_group_rules.api = mock.Mock()
        self.security_group_rules.api.client = mock.Mock()
        self.security_group_rules.api.client.delete = mock.Mock(return_value=(resp, body))
        self.security_group_rules.delete(self.id)
        resp.status_code = 500
        self.assertRaises(Exception, self.security_group_rules.delete, self.id)