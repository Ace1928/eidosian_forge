from keystoneauth1 import exceptions as ksa_exceptions
import testresources
from testtools import matchers
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient import utils
class FindResourceTestCase(test_utils.TestCase):

    def setUp(self):
        super(FindResourceTestCase, self).setUp()
        self.manager = FakeManager()

    def test_find_none(self):
        self.assertRaises(ksc_exceptions.CommandError, utils.find_resource, self.manager, 'asdf')

    def test_find_by_integer_id(self):
        output = utils.find_resource(self.manager, 1234)
        self.assertEqual(output, self.manager.resources['1234'])

    def test_find_by_str_id(self):
        output = utils.find_resource(self.manager, '1234')
        self.assertEqual(output, self.manager.resources['1234'])

    def test_find_by_uuid(self):
        uuid = '8e8ec658-c7b0-4243-bdf8-6f7f2952c0d0'
        output = utils.find_resource(self.manager, uuid)
        self.assertEqual(output, self.manager.resources[uuid])

    def test_find_by_unicode(self):
        name = 'ã\x82½test'
        output = utils.find_resource(self.manager, name)
        self.assertEqual(output, self.manager.resources[name])

    def test_find_by_str_name(self):
        output = utils.find_resource(self.manager, 'entity_one')
        self.assertEqual(output, self.manager.resources['1234'])

    def test_find_by_int_name(self):
        output = utils.find_resource(self.manager, 9876)
        self.assertEqual(output, self.manager.resources['5678'])

    def test_find_no_unique_match(self):
        self.assertRaises(ksc_exceptions.CommandError, utils.find_resource, self.manager, 9999)