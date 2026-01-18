import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
class CliDomainConfigAllTestCase(unit.SQLDriverOverrides, unit.TestCase):

    def setUp(self):
        self.useFixture(database.Database())
        super(CliDomainConfigAllTestCase, self).setUp()
        self.load_backends()
        self.config_fixture.config(group='identity', domain_config_dir=unit.TESTCONF + '/domain_configs_multi_ldap')
        self.domain_count = 3
        self.setup_initial_domains()
        self.logging = self.useFixture(fixtures.FakeLogger(level=logging.INFO))

    def config_files(self):
        self.config_fixture.register_cli_opt(cli.command_opt)
        config_files = super(CliDomainConfigAllTestCase, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files

    def cleanup_domains(self):
        for domain in self.domains:
            if domain == 'domain_default':
                PROVIDERS.domain_config_api.delete_config(CONF.identity.default_domain_id)
                continue
            this_domain = self.domains[domain]
            this_domain['enabled'] = False
            PROVIDERS.resource_api.update_domain(this_domain['id'], this_domain)
            PROVIDERS.resource_api.delete_domain(this_domain['id'])
        self.domains = {}

    def config(self, config_files):
        CONF(args=['domain_config_upload', '--all'], project='keystone', default_config_files=config_files)

    def setup_initial_domains(self):

        def create_domain(domain):
            return PROVIDERS.resource_api.create_domain(domain['id'], domain)
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        self.domains = {}
        self.addCleanup(self.cleanup_domains)
        for x in range(1, self.domain_count):
            domain = 'domain%s' % x
            self.domains[domain] = create_domain({'id': uuid.uuid4().hex, 'name': domain})
        self.default_domain = unit.new_domain_ref(description=u'The default domain', id=CONF.identity.default_domain_id, name=u'Default')
        self.domains['domain_default'] = create_domain(self.default_domain)

    def test_config_upload(self):
        default_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap'}}
        domain1_config = {'ldap': {'url': 'fake://memory1', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap', 'list_limit': '101'}}
        domain2_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=myroot,cn=com', 'group_tree_dn': 'ou=UserGroups,dc=myroot,dc=org', 'user_tree_dn': 'ou=Users,dc=myroot,dc=org'}, 'identity': {'driver': 'ldap'}}
        provider_api.ProviderAPIs._clear_registry_instances()
        cli.DomainConfigUpload.main()
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(CONF.identity.default_domain_id)
        self.assertEqual(default_config, res)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domains['domain1']['id'])
        self.assertEqual(domain1_config, res)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domains['domain2']['id'])
        self.assertEqual(domain2_config, res)