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
class TestImageStorageUsage(base.TestDriver, base.FunctionalInitWrapper):

    def setUp(self):
        db_tests.load(get_db, reset_db)
        super(TestImageStorageUsage, self).setUp()
        self.addCleanup(db_tests.reset)
        self.contexts = {}
        for owner in (uuids.owner1, uuids.owner2):
            ctxt = glance_context.RequestContext(project_id=owner)
            self.contexts[owner] = ctxt
            statuses = ['queued', 'active', 'uploading', 'importing', 'deleted']
            for status in statuses:
                for num in range(0, 2):
                    size = statuses.index(status) * 100
                    image = self.db_api.image_create(ctxt, {'status': status, 'owner': owner, 'size': size, 'name': 'test-%s-%i' % (status, num)})
                    if status == 'active':
                        loc_status = num == 0 and 'active' or 'deleted'
                        self.db_api.image_location_add(ctxt, image['id'], {'url': 'foo://bar', 'metadata': {}, 'status': loc_status})
                        self.db_api.image_set_property_atomic(image['id'], 'os_glance_importing_to_stores', num == 0 and 'fakestore' or '')

    def test_get_staging_usage(self):
        for owner, ctxt in self.contexts.items():
            usage = self.db_api.user_get_staging_usage(ctxt, ctxt.owner)
            self.assertEqual(1100, usage)

    def test_get_storage_usage(self):
        for owner, ctxt in self.contexts.items():
            usage = self.db_api.user_get_storage_usage(ctxt, ctxt.owner)
            self.assertEqual(100, usage)

    def test_get_image_count(self):
        for owner, ctxt in self.contexts.items():
            count = self.db_api.user_get_image_count(ctxt, ctxt.owner)
            self.assertEqual(8, count)

    def test_get_uploading_count(self):
        for owner, ctxt in self.contexts.items():
            count = self.db_api.user_get_uploading_count(ctxt, ctxt.owner)
            self.assertEqual(5, count)