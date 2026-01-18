import datetime
from unittest import mock
import uuid
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import revoke_model
from keystone.revoke.backends import sql
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_backend_sql
from keystone.token import provider
def _sample_blank_token():
    issued_delta = datetime.timedelta(minutes=-2)
    issued_at = timeutils.utcnow() + issued_delta
    token_data = revoke_model.blank_token_data(issued_at)
    return token_data