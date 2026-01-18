import fixtures
from keystone.common import fernet_utils as utils
class KeyRepository(fixtures.Fixture):

    def __init__(self, config_fixture, key_group, max_active_keys):
        super(KeyRepository, self).__init__()
        self.config_fixture = config_fixture
        self.max_active_keys = max_active_keys
        self.key_group = key_group

    def setUp(self):
        super(KeyRepository, self).setUp()
        directory = self.useFixture(fixtures.TempDir()).path
        self.config_fixture.config(group=self.key_group, key_repository=directory)
        fernet_utils = utils.FernetUtils(directory, self.max_active_keys, self.key_group)
        fernet_utils.create_key_directory()
        fernet_utils.initialize_key_repository()