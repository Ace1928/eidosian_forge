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
class TestTrustFlush(unit.SQLDriverOverrides, unit.BaseTestCase):

    class FakeConfCommand(object):

        def __init__(self, parent):
            self.extension = False
            self.project_id = parent.command_project_id
            self.trustor_user_id = parent.command_trustor_user_id
            self.trustee_user_id = parent.command_trustee_user_id
            self.date = parent.command_date

    def setUp(self):
        super(TestTrustFlush, self).setUp()
        self.useFixture(database.Database())
        self.config_fixture = self.useFixture(oslo_config.fixture.Config(CONF))
        self.config_fixture.register_cli_opt(cli.command_opt)
        parser_test = argparse.ArgumentParser()
        subparsers = parser_test.add_subparsers()
        self.parser = cli.TrustFlush.add_argument_parser(subparsers)

    def config_files(self):
        config_files = super(TestTrustFlush, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files

    def test_trust_flush(self):
        self.command_project_id = None
        self.command_trustor_user_id = None
        self.command_trustee_user_id = None
        self.command_date = datetime.datetime.utcnow()
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))

        def fake_load_backends():
            return dict(trust_api=keystone.trust.core.Manager())
        self.useFixture(fixtures.MockPatch('keystone.server.backends.load_backends', side_effect=fake_load_backends))
        trust = cli.TrustFlush()
        trust.main()

    def test_trust_flush_with_invalid_date(self):
        self.command_project_id = None
        self.command_trustor_user_id = None
        self.command_trustee_user_id = None
        self.command_date = '4/10/92'
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))

        def fake_load_backends():
            return dict(trust_api=keystone.trust.core.Manager())
        self.useFixture(fixtures.MockPatch('keystone.server.backends.load_backends', side_effect=fake_load_backends))
        provider_api.ProviderAPIs._clear_registry_instances()
        trust = cli.TrustFlush()
        self.assertRaises(ValueError, trust.main)