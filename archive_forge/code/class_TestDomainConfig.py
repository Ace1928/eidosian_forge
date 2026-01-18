from openstack.identity.v3 import domain_config
from openstack.tests.unit import base
class TestDomainConfig(base.TestCase):

    def test_basic(self):
        sot = domain_config.DomainConfig()
        self.assertEqual('config', sot.resource_key)
        self.assertEqual('/domains/%(domain_id)s/config', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)

    def test_make_it(self):
        sot = domain_config.DomainConfig(**EXAMPLE)
        self.assertIsInstance(sot.identity, domain_config.DomainConfigDriver)
        self.assertEqual(EXAMPLE['identity']['driver'], sot.identity.driver)
        self.assertIsInstance(sot.ldap, domain_config.DomainConfigLDAP)
        self.assertEqual(EXAMPLE['ldap']['url'], sot.ldap.url)
        self.assertEqual(EXAMPLE['ldap']['user_tree_dn'], sot.ldap.user_tree_dn)