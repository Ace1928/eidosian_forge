import itertools
import os
from unittest import mock
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
class TestDomainConfigs(unit.BaseTestCase):

    def setUp(self):
        super(TestDomainConfigs, self).setUp()
        self.addCleanup(CONF.reset)
        self.tmp_dir = unit.dirs.tmp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(domain_config_dir=self.tmp_dir, group='identity')

    def test_config_for_nonexistent_domain(self):
        """Having a config for a non-existent domain will be ignored.

        There are no assertions in this test because there are no side
        effects. If there is a config file for a domain that does not
        exist it should be ignored.

        """
        domain_id = uuid.uuid4().hex
        domain_config_filename = os.path.join(self.tmp_dir, 'keystone.%s.conf' % domain_id)
        self.addCleanup(lambda: os.remove(domain_config_filename))
        with open(domain_config_filename, 'w'):
            'Write an empty config file.'
        e = exception.DomainNotFound(domain_id=domain_id)
        mock_assignment_api = mock.Mock()
        mock_assignment_api.get_domain_by_name.side_effect = e
        domain_config = identity.DomainConfigs()
        fake_standard_driver = None
        domain_config.setup_domain_drivers(fake_standard_driver, mock_assignment_api)

    def test_config_for_dot_name_domain(self):
        domain_config_filename = os.path.join(self.tmp_dir, 'keystone.abc.def.com.conf')
        with open(domain_config_filename, 'w'):
            'Write an empty config file.'
        self.addCleanup(os.remove, domain_config_filename)
        with mock.patch.object(identity.DomainConfigs, '_load_config_from_file') as mock_load_config:
            domain_config = identity.DomainConfigs()
            fake_assignment_api = None
            fake_standard_driver = None
            domain_config.setup_domain_drivers(fake_standard_driver, fake_assignment_api)
            mock_load_config.assert_called_once_with(fake_assignment_api, [domain_config_filename], 'abc.def.com')

    def test_config_for_multiple_sql_backend(self):
        domains_config = identity.DomainConfigs()
        drivers = []
        files = []
        for idx, is_sql in enumerate((True, False, True)):
            drv = mock.Mock(is_sql=is_sql)
            drivers.append(drv)
            name = 'dummy.{0}'.format(idx)
            files.append(''.join((identity.DOMAIN_CONF_FHEAD, name, identity.DOMAIN_CONF_FTAIL)))

        def walk_fake(*a, **kwa):
            return (('/fake/keystone/domains/config', [], files),)
        generic_driver = mock.Mock(is_sql=False)
        assignment_api = mock.Mock()
        id_factory = itertools.count()
        assignment_api.get_domain_by_name.side_effect = lambda name: {'id': next(id_factory), '_': 'fake_domain'}
        load_driver_mock = mock.Mock(side_effect=drivers)
        with mock.patch.object(os, 'walk', walk_fake):
            with mock.patch.object(identity.cfg, 'ConfigOpts'):
                with mock.patch.object(domains_config, '_load_driver', load_driver_mock):
                    self.assertRaises(exception.MultipleSQLDriversInConfig, domains_config.setup_domain_drivers, generic_driver, assignment_api)
                    self.assertEqual(3, load_driver_mock.call_count)