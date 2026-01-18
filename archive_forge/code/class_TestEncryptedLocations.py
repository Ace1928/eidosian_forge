import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestEncryptedLocations(test_utils.BaseTestCase):

    def setUp(self):
        super(TestEncryptedLocations, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.context = glance.context.RequestContext(user=USER1, tenant=TENANT1)
        self.image_repo = glance.db.ImageRepo(self.context, self.db)
        self.image_factory = glance.domain.ImageFactory()
        self.crypt_key = '0123456789abcdef'
        self.config(metadata_encryption_key=self.crypt_key)
        self.foo_bar_location = [{'url': 'foo', 'metadata': {}, 'status': 'active'}, {'url': 'bar', 'metadata': {}, 'status': 'active'}]

    def test_encrypt_locations_on_add(self):
        image = self.image_factory.new_image(UUID1)
        image.locations = self.foo_bar_location
        self.image_repo.add(image)
        db_data = self.db.image_get(self.context, UUID1)
        self.assertNotEqual(db_data['locations'], ['foo', 'bar'])
        decrypted_locations = [crypt.urlsafe_decrypt(self.crypt_key, location['url']) for location in db_data['locations']]
        self.assertEqual([location['url'] for location in self.foo_bar_location], decrypted_locations)

    def test_encrypt_locations_on_save(self):
        image = self.image_factory.new_image(UUID1)
        self.image_repo.add(image)
        image.locations = self.foo_bar_location
        self.image_repo.save(image)
        db_data = self.db.image_get(self.context, UUID1)
        self.assertNotEqual(db_data['locations'], ['foo', 'bar'])
        decrypted_locations = [crypt.urlsafe_decrypt(self.crypt_key, location['url']) for location in db_data['locations']]
        self.assertEqual([location['url'] for location in self.foo_bar_location], decrypted_locations)

    def test_decrypt_locations_on_get(self):
        url_loc = ['ping', 'pong']
        orig_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in url_loc]
        encrypted_locs = [crypt.urlsafe_encrypt(self.crypt_key, location) for location in url_loc]
        encrypted_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in encrypted_locs]
        self.assertNotEqual(encrypted_locations, orig_locations)
        db_data = _db_fixture(UUID1, owner=TENANT1, locations=encrypted_locations)
        self.db.image_create(None, db_data)
        image = self.image_repo.get(UUID1)
        self.assertIn('id', image.locations[0])
        self.assertIn('id', image.locations[1])
        image.locations[0].pop('id')
        image.locations[1].pop('id')
        self.assertEqual(orig_locations, image.locations)

    def test_decrypt_locations_on_list(self):
        url_loc = ['ping', 'pong']
        orig_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in url_loc]
        encrypted_locs = [crypt.urlsafe_encrypt(self.crypt_key, location) for location in url_loc]
        encrypted_locations = [{'url': location, 'metadata': {}, 'status': 'active'} for location in encrypted_locs]
        self.assertNotEqual(encrypted_locations, orig_locations)
        db_data = _db_fixture(UUID1, owner=TENANT1, locations=encrypted_locations)
        self.db.image_create(None, db_data)
        image = self.image_repo.list()[0]
        self.assertIn('id', image.locations[0])
        self.assertIn('id', image.locations[1])
        image.locations[0].pop('id')
        image.locations[1].pop('id')
        self.assertEqual(orig_locations, image.locations)