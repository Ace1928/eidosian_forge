from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
@db_api.retry_db_errors
def _decorated_function(self, fail_count, exc_to_raise):
    self.fail_count = getattr(self, 'fail_count', fail_count + 1) - 1
    if self.fail_count:
        raise exc_to_raise