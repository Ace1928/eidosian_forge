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
class FernetUtilsTestCase(unit.BaseTestCase):

    def setUp(self):
        super(FernetUtilsTestCase, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))

    def test_debug_message_logged_when_loading_fernet_token_keys(self):
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))
        logging_fixture = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        fernet_utilities = fernet_utils.FernetUtils(CONF.fernet_tokens.key_repository, CONF.fernet_tokens.max_active_keys, 'fernet_tokens')
        fernet_utilities.load_keys()
        expected_debug_message = 'Loaded 2 Fernet keys from %(dir)s, but `[fernet_tokens] max_active_keys = %(max)d`; perhaps there have not been enough key rotations to reach `max_active_keys` yet?' % {'dir': CONF.fernet_tokens.key_repository, 'max': CONF.fernet_tokens.max_active_keys}
        self.assertIn(expected_debug_message, logging_fixture.output)

    def test_debug_message_not_logged_when_loading_fernet_credential_key(self):
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'credential', CONF.fernet_tokens.max_active_keys))
        logging_fixture = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        fernet_utilities = fernet_utils.FernetUtils(CONF.credential.key_repository, credential_fernet.MAX_ACTIVE_KEYS, 'credential')
        fernet_utilities.load_keys()
        debug_message = 'Loaded 2 Fernet keys from %(dir)s, but `[credential] max_active_keys = %(max)d`; perhaps there have not been enough key rotations to reach `max_active_keys` yet?' % {'dir': CONF.credential.key_repository, 'max': credential_fernet.MAX_ACTIVE_KEYS}
        self.assertNotIn(debug_message, logging_fixture.output)