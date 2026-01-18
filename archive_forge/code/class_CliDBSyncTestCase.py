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
class CliDBSyncTestCase(unit.BaseTestCase):

    class FakeConfCommand(object):

        def __init__(self, parent):
            self.extension = False
            self.check = parent.command_check
            self.expand = parent.command_expand
            self.migrate = parent.command_migrate
            self.contract = parent.command_contract
            self.version = None

    def setUp(self):
        super().setUp()
        self.config_fixture = self.useFixture(oslo_config.fixture.Config(CONF))
        self.config_fixture.register_cli_opt(cli.command_opt)
        self.patchers = patchers = [mock.patch.object(upgrades, 'offline_sync_database_to_version'), mock.patch.object(upgrades, 'expand_schema'), mock.patch.object(upgrades, 'migrate_data'), mock.patch.object(upgrades, 'contract_schema')]
        for p in patchers:
            p.start()
        self.command_check = False
        self.command_expand = False
        self.command_migrate = False
        self.command_contract = False

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        super().tearDown()

    def _assert_correct_call(self, mocked_function):
        for func in [upgrades.offline_sync_database_to_version, upgrades.expand_schema, upgrades.migrate_data, upgrades.contract_schema]:
            if func == mocked_function:
                self.assertTrue(func.called)
            else:
                self.assertFalse(func.called)

    def test_db_sync(self):
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        cli.DbSync.main()
        self._assert_correct_call(upgrades.offline_sync_database_to_version)

    def test_db_sync_expand(self):
        self.command_expand = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        cli.DbSync.main()
        self._assert_correct_call(upgrades.expand_schema)

    def test_db_sync_migrate(self):
        self.command_migrate = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        cli.DbSync.main()
        self._assert_correct_call(upgrades.migrate_data)

    def test_db_sync_contract(self):
        self.command_contract = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        cli.DbSync.main()
        self._assert_correct_call(upgrades.contract_schema)