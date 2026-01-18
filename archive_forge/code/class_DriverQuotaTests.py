import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class DriverQuotaTests(test_utils.BaseTestCase):

    def setUp(self):
        super(DriverQuotaTests, self).setUp()
        self.owner_id1 = str(uuid.uuid4())
        self.context1 = context.RequestContext(is_admin=False, user=self.owner_id1, tenant=self.owner_id1, auth_token='%s:%s:user' % (self.owner_id1, self.owner_id1))
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)
        dt1 = timeutils.utcnow()
        dt2 = dt1 + datetime.timedelta(microseconds=5)
        fixtures = [{'id': UUID1, 'created_at': dt1, 'updated_at': dt1, 'size': 13, 'owner': self.owner_id1}, {'id': UUID2, 'created_at': dt1, 'updated_at': dt2, 'size': 17, 'owner': self.owner_id1}, {'id': UUID3, 'created_at': dt2, 'updated_at': dt2, 'size': 7, 'owner': self.owner_id1}]
        self.owner1_fixtures = [build_image_fixture(**fixture) for fixture in fixtures]
        for fixture in self.owner1_fixtures:
            self.db_api.image_create(self.context1, fixture)

    def test_storage_quota(self):
        total = functools.reduce(lambda x, y: x + y, [f['size'] for f in self.owner1_fixtures])
        x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1)
        self.assertEqual(total, x)

    def test_storage_quota_without_image_id(self):
        total = functools.reduce(lambda x, y: x + y, [f['size'] for f in self.owner1_fixtures])
        total = total - self.owner1_fixtures[0]['size']
        x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1, image_id=self.owner1_fixtures[0]['id'])
        self.assertEqual(total, x)

    def test_storage_quota_multiple_locations(self):
        dt1 = timeutils.utcnow()
        sz = 53
        new_fixture_dict = {'id': str(uuid.uuid4()), 'created_at': dt1, 'updated_at': dt1, 'size': sz, 'owner': self.owner_id1}
        new_fixture = build_image_fixture(**new_fixture_dict)
        new_fixture['locations'].append({'url': 'file:///some/path/file', 'metadata': {}, 'status': 'active'})
        self.db_api.image_create(self.context1, new_fixture)
        total = functools.reduce(lambda x, y: x + y, [f['size'] for f in self.owner1_fixtures]) + sz * 2
        x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1)
        self.assertEqual(total, x)

    def test_storage_quota_deleted_image(self):
        dt1 = timeutils.utcnow()
        sz = 53
        image_id = str(uuid.uuid4())
        new_fixture_dict = {'id': image_id, 'created_at': dt1, 'updated_at': dt1, 'size': sz, 'owner': self.owner_id1}
        new_fixture = build_image_fixture(**new_fixture_dict)
        new_fixture['locations'].append({'url': 'file:///some/path/file', 'metadata': {}, 'status': 'active'})
        self.db_api.image_create(self.context1, new_fixture)
        total = functools.reduce(lambda x, y: x + y, [f['size'] for f in self.owner1_fixtures])
        x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1)
        self.assertEqual(total + sz * 2, x)
        self.db_api.image_destroy(self.context1, image_id)
        x = self.db_api.user_get_storage_usage(self.context1, self.owner_id1)
        self.assertEqual(total, x)