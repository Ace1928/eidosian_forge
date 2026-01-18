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
class TestSqlAlchemyDriver(base.TestDriver, base.DriverTests, base.FunctionalInitWrapper):

    def setUp(self):
        db_tests.load(get_db, reset_db)
        super(TestSqlAlchemyDriver, self).setUp()
        self.addCleanup(db_tests.reset)

    def test_get_image_with_invalid_long_image_id(self):
        image_id = '343f9ba5-0197-41be-9543-16bbb32e12aa-xxxxxx'
        self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, image_id)

    def test_image_tag_delete_with_invalid_long_image_id(self):
        image_id = '343f9ba5-0197-41be-9543-16bbb32e12aa-xxxxxx'
        self.assertRaises(exception.NotFound, self.db_api.image_tag_delete, self.context, image_id, 'fake')

    def test_image_tag_get_all_with_invalid_long_image_id(self):
        image_id = '343f9ba5-0197-41be-9543-16bbb32e12aa-xxxxxx'
        self.assertRaises(exception.NotFound, self.db_api.image_tag_get_all, self.context, image_id)

    def test_user_get_storage_usage_with_invalid_long_image_id(self):
        image_id = '343f9ba5-0197-41be-9543-16bbb32e12aa-xxxxxx'
        self.assertRaises(exception.NotFound, self.db_api.user_get_storage_usage, self.context, 'fake_owner_id', image_id)