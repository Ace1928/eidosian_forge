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
class CliLoggingTestCase(unit.BaseTestCase):

    def setUp(self):
        self.config_fixture = self.useFixture(oslo_config.fixture.Config(CONF))
        self.config_fixture.register_cli_opt(cli.command_opt)
        self.useFixture(fixtures.MockPatch('oslo_config.cfg.find_config_files', return_value=[]))
        fd = self.useFixture(temporaryfile.SecureTempFile())
        self.fake_config_file = fd.file_name
        super(CliLoggingTestCase, self).setUp()

        class FakeConfCommand(object):

            def __init__(self):
                self.cmd_class = mock.Mock()
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', FakeConfCommand()))
        self.logging = self.useFixture(fixtures.FakeLogger(level=log.WARN))

    def test_absent_config_logs_warning(self):
        expected_msg = 'Config file not found, using default configs.'
        cli.main(argv=['keystone-manage', 'db_sync'])
        self.assertThat(self.logging.output, matchers.Contains(expected_msg))

    def test_present_config_does_not_log_warning(self):
        fake_argv = ['keystone-manage', '--config-file', self.fake_config_file, 'doctor']
        cli.main(argv=fake_argv)
        expected_msg = 'Config file not found, using default configs.'
        self.assertNotIn(expected_msg, self.logging.output)