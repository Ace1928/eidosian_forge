import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _create_tasks(self):
    now = timeutils.utcnow()
    times = [now + datetime.timedelta(seconds=5 * i) for i in range(4)]
    self.tasks = [_db_fixture(UUID1, owner=TENANT1, created_at=times[0], updated_at=times[0]), _db_fixture(UUID2, owner=TENANT2, type='import', created_at=times[1], updated_at=times[1]), _db_fixture(UUID3, owner=TENANT3, type='import', created_at=times[2], updated_at=times[2]), _db_fixture(UUID4, owner=TENANT4, type='import', created_at=times[3], updated_at=times[3])]
    [self.db.task_create(None, task) for task in self.tasks]