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
class TestDriver(test_utils.BaseTestCase):

    def setUp(self):
        super(TestDriver, self).setUp()
        context_cls = context.RequestContext
        self.adm_context = context_cls(is_admin=True, auth_token='user:user:admin')
        self.context = context_cls(is_admin=False, auth_token='user:user:user')
        self.db_api = db_tests.get_db(self.config)
        db_tests.reset_db(self.db_api)
        self.fixtures = self.build_image_fixtures()
        self.create_images(self.fixtures)

    def build_image_fixtures(self):
        dt1 = timeutils.utcnow()
        dt2 = dt1 + datetime.timedelta(microseconds=5)
        fixtures = [{'id': UUID1, 'created_at': dt1, 'updated_at': dt1, 'properties': {'foo': 'bar', 'far': 'boo'}, 'protected': True, 'size': 13}, {'id': UUID2, 'created_at': dt1, 'updated_at': dt2, 'size': 17}, {'id': UUID3, 'created_at': dt2, 'updated_at': dt2}]
        return [build_image_fixture(**fixture) for fixture in fixtures]

    def create_images(self, images):
        for fixture in images:
            self.db_api.image_create(self.adm_context, fixture)
            self.delay_inaccurate_clock()