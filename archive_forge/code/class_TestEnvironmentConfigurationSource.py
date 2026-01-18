import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
class TestEnvironmentConfigurationSource(base.BaseTestCase):

    def setUp(self):
        super(TestEnvironmentConfigurationSource, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.conf_fixture = self.useFixture(fixture.Config(self.conf))
        self.conf.register_opt(cfg.StrOpt('bar'), 'foo')
        self.conf.register_opt(cfg.StrOpt('baz', regex='^[a-z].*$'), 'foo')

        def cleanup():
            for env in ('OS_FOO__BAR', 'OS_FOO__BAZ'):
                if env in os.environ:
                    del os.environ[env]
        self.addCleanup(cleanup)

    def test_simple_environment_get(self):
        self.conf(args=[])
        env_value = 'goodbye'
        os.environ['OS_FOO__BAR'] = env_value
        self.assertEqual(env_value, self.conf['foo']['bar'])

    def test_env_beats_files(self):
        file_value = 'hello'
        env_value = 'goodbye'
        self.conf(args=[])
        self.conf_fixture.load_raw_values(group='foo', bar=file_value)
        self.assertEqual(file_value, self.conf['foo']['bar'])
        self.conf.reload_config_files()
        os.environ['OS_FOO__BAR'] = env_value
        self.assertEqual(env_value, self.conf['foo']['bar'])

    def test_cli_beats_env(self):
        env_value = 'goodbye'
        cli_value = 'cli'
        os.environ['OS_FOO__BAR'] = env_value
        self.conf.register_cli_opt(cfg.StrOpt('bar'), 'foo')
        self.conf(args=['--foo=%s' % cli_value])
        self.assertEqual(cli_value, self.conf['foo']['bar'])

    def test_use_env_false_allows_files(self):
        file_value = 'hello'
        env_value = 'goodbye'
        os.environ['OS_FOO__BAR'] = env_value
        self.conf(args=[], use_env=False)
        self.conf_fixture.load_raw_values(group='foo', bar=file_value)
        self.assertEqual(file_value, self.conf['foo']['bar'])
        self.conf.reset()
        self.conf(args=[], use_env=True)
        self.assertEqual(env_value, self.conf['foo']['bar'])

    def test_invalid_env(self):
        self.conf(args=[])
        env_value = 'ABC'
        os.environ['OS_FOO__BAZ'] = env_value
        with testtools.ExpectedException(cfg.ConfigSourceValueError):
            self.conf['foo']['baz']