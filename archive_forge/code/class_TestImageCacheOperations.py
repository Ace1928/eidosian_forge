from oslo_config import cfg
from oslo_db import options
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context as glance_context
import glance.db.sqlalchemy.api
from glance.db.sqlalchemy import models as db_models
from glance.db.sqlalchemy import models_metadef as metadef_models
import glance.tests.functional.db as db_tests
from glance.tests.functional.db import base
from glance.tests.functional.db import base_metadef
class TestImageCacheOperations(base.TestDriver, base.FunctionalInitWrapper):

    def setUp(self):
        db_tests.load(get_db, reset_db)
        super(TestImageCacheOperations, self).setUp()
        self.addCleanup(db_tests.reset)
        self.images = []
        for num in range(0, 2):
            size = 100
            image = self.db_api.image_create(self.adm_context, {'status': 'active', 'owner': self.adm_context.owner, 'size': size, 'name': 'test-%s-%i' % ('active', num)})
            self.images.append(image)
        self.node_references = [self.db_api.node_reference_create(self.adm_context, 'node_url_1'), self.db_api.node_reference_create(self.adm_context, 'node_url_2')]
        for node in self.node_references:
            if node['node_reference_url'] == 'node_url_2':
                continue
            for image in self.images:
                self.db_api.insert_cache_details(self.adm_context, 'node_url_1', image['id'], image['size'], hits=3)

    def test_node_reference_get_by_url(self):
        node_reference = self.db_api.node_reference_get_by_url(self.adm_context, 'node_url_1')
        self.assertEqual('node_url_1', node_reference['node_reference_url'])

    def test_node_reference_get_by_url_not_found(self):
        self.assertRaises(exception.NotFound, self.db_api.node_reference_get_by_url, self.adm_context, 'garbage_url')

    def test_get_cached_images(self):
        cached_images = self.db_api.get_cached_images(self.adm_context, 'node_url_1')
        self.assertEqual(2, len(cached_images))
        cached_images = self.db_api.get_cached_images(self.adm_context, 'node_url_2')
        self.assertEqual(0, len(cached_images))

    def test_get_hit_count(self):
        hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
        self.assertEqual(3, hit_count)
        hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_2')
        self.assertEqual(0, hit_count)

    def test_delete_all_cached_images(self):
        self.db_api.delete_all_cached_images(self.adm_context, 'node_url_1')
        cached_images = self.db_api.get_cached_images(self.adm_context, 'node_url_1')
        self.assertEqual(0, len(cached_images))

    def test_delete_cached_image(self):
        self.db_api.delete_cached_image(self.adm_context, self.images[0]['id'], 'node_url_1')
        self.assertFalse(self.db_api.is_image_cached_for_node(self.adm_context, 'node_url_1', self.images[0]['id']))

    def test_get_least_recently_accessed(self):
        recently_accessed = self.db_api.get_least_recently_accessed(self.adm_context, 'node_url_1')
        self.assertEqual(self.images[0]['id'], recently_accessed)

    def test_is_image_cached_for_node(self):
        self.assertTrue(self.db_api.is_image_cached_for_node(self.adm_context, 'node_url_1', self.images[0]['id']))
        self.assertFalse(self.db_api.is_image_cached_for_node(self.adm_context, 'node_url_2', self.images[0]['id']))

    def test_update_hit_count(self):
        hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
        self.assertEqual(3, hit_count)
        self.db_api.update_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
        hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
        self.assertEqual(4, hit_count)