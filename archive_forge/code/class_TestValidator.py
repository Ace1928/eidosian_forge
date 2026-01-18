from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import validator
class TestValidator(base.BaseTestCase):

    def setUp(self):
        super(TestValidator, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.conf_fixture = self.useFixture(fixture.Config(self.conf))
        validator._register_cli_opts(self.conf)

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_passing(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf')
        m = mock.mock_open(read_data=VALID_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(0, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_deprecated(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf')
        m = mock.mock_open(read_data=DEPRECATED_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(0, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_deprecated_fatal_warnings(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf', fatal_warnings=True)
        m = mock.mock_open(read_data=DEPRECATED_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(1, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_missing(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf')
        m = mock.mock_open(read_data=INVALID_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(1, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_missing_group(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf')
        m = mock.mock_open(read_data=MISSING_GROUP_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(1, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_exclude_groups(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf', exclude_group=['oo'])
        m = mock.mock_open(read_data=MISSING_GROUP_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(0, validator._validate(self.conf))

    @mock.patch('oslo_config.validator.load_opt_data')
    def test_check_defaults(self, mock_lod):
        mock_lod.return_value = OPT_DATA
        self.conf_fixture.config(opt_data='mocked.yaml', input_file='mocked.conf', check_defaults=True)
        m = mock.mock_open(read_data=VALID_CONF)
        with mock.patch('builtins.open', m):
            self.assertEqual(0, validator._validate(self.conf))

    def test_invalid_options(self):
        self.assertRaises(RuntimeError, validator._validate, self.conf)