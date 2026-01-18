import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
class ServiceHelperTests(unit.BaseTestCase):

    @application.fail_gracefully
    def _do_test(self):
        raise Exception('Test Exc')

    def test_fail_gracefully(self):
        self.assertRaises(unit.UnexpectedExit, self._do_test)