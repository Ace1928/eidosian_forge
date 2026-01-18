import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestServersTagsV226(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.26'

    def _boot_server_with_tags(self, tags=['t1', 't2']):
        uuid = self._create_server().id
        self.client.servers.set_tags(uuid, tags)
        return uuid

    def test_show(self):
        uuid = self._boot_server_with_tags()
        output = self.nova('show %s' % uuid)
        self.assertEqual('["t1", "t2"]', self._get_value_from_the_table(output, 'tags'))

    def test_unicode_tag_correctly_displayed(self):
        """Regression test for bug #1669683.

        List and dict fields with unicode cannot be correctly
        displayed.

        Ensure that once we fix this it doesn't regress.
        """
        uuid = self._boot_server_with_tags(tags=['中文标签'])
        output = self.nova('show %s' % uuid)
        self.assertEqual('["中文标签"]', self._get_value_from_the_table(output, 'tags'))

    def test_list(self):
        uuid = self._boot_server_with_tags()
        output = self.nova('server-tag-list %s' % uuid)
        tags = self._get_list_of_values_from_single_column_table(output, 'Tag')
        self.assertEqual(['t1', 't2'], tags)

    def test_add(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-add %s t3' % uuid)
        self.assertEqual(['t1', 't2', 't3'], self.client.servers.tag_list(uuid))

    def test_add_many(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-add %s t3 t4' % uuid)
        self.assertEqual(['t1', 't2', 't3', 't4'], self.client.servers.tag_list(uuid))

    def test_set(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-set %s t3 t4' % uuid)
        self.assertEqual(['t3', 't4'], self.client.servers.tag_list(uuid))

    def test_delete(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-delete %s t2' % uuid)
        self.assertEqual(['t1'], self.client.servers.tag_list(uuid))

    def test_delete_many(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-delete %s t1 t2' % uuid)
        self.assertEqual([], self.client.servers.tag_list(uuid))

    def test_delete_all(self):
        uuid = self._boot_server_with_tags()
        self.nova('server-tag-delete-all %s' % uuid)
        self.assertEqual([], self.client.servers.tag_list(uuid))