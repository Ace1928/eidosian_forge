from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
class TestTld(BaseDesignateTest):

    def setUp(self):
        super().setUp()
        tld_name = random_tld()
        self.tld = self.useFixture(TLDFixture(name=tld_name, description='A random tld')).tld
        self.assertEqual(self.tld.name, tld_name)
        self.assertEqual(self.tld.description, 'A random tld')

    def test_tld_list(self):
        tlds = self.clients.as_user('admin').tld_list()
        self.assertGreater(len(tlds), 0)

    def test_tld_create_and_show(self):
        tld = self.clients.as_user('admin').tld_show(self.tld.id)
        self.assertEqual(tld.name, self.tld.name)
        self.assertEqual(tld.created_at, self.tld.created_at)
        self.assertEqual(tld.id, self.tld.id)
        self.assertEqual(tld.name, self.tld.name)
        self.assertEqual(tld.updated_at, self.tld.updated_at)

    def test_tld_delete(self):
        client = self.clients.as_user('admin')
        client.tld_delete(self.tld.id)
        self.assertRaises(CommandFailed, client.tld_show, self.tld.id)

    def test_tld_set(self):
        client = self.clients.as_user('admin')
        updated_name = random_tld('updated')
        tld = client.tld_set(self.tld.id, name=updated_name, description='An updated tld')
        self.assertEqual(tld.description, 'An updated tld')
        self.assertEqual(tld.name, updated_name)

    def test_tld_set_no_description(self):
        client = self.clients.as_user('admin')
        tld = client.tld_set(self.tld.id, no_description=True)
        self.assertEqual(tld.description, 'None')

    def test_no_set_tld_with_description_and_no_description(self):
        client = self.clients.as_user('admin')
        self.assertRaises(CommandFailed, client.tld_set, self.tld.id, description='An updated tld', no_description=True)