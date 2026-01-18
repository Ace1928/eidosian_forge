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
class TestMappingEngineTester(unit.BaseTestCase):

    class FakeConfCommand(object):

        def __init__(self, parent):
            self.extension = False
            self.rules = parent.command_rules
            self.input = parent.command_input
            self.prefix = parent.command_prefix
            self.engine_debug = parent.command_engine_debug
            self.mapping_schema_version = parent.mapping_schema_version

    def setUp(self):
        super(TestMappingEngineTester, self).setUp()
        self.mapping_id = uuid.uuid4().hex
        self.rules_pathname = None
        self.rules = None
        self.assertion_pathname = None
        self.assertion = None
        self.logging = self.useFixture(fixtures.LoggerFixture())
        self.useFixture(database.Database())
        self.config_fixture = self.useFixture(oslo_config.fixture.Config(CONF))
        self.config_fixture.register_cli_opt(cli.command_opt)
        parser_test = argparse.ArgumentParser()
        subparsers = parser_test.add_subparsers()
        self.parser = cli.MappingEngineTester.add_argument_parser(subparsers)
        self.mapping_schema_version = '1.0'

    def config_files(self):
        config_files = super(TestMappingEngineTester, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files

    def test_mapping_engine_tester_with_invalid_rules_file(self):
        tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
        tmpinvalidfile = tempfilejson.file_name
        with open(tmpinvalidfile, 'w') as f:
            f.write('This is an invalid data')
        self.command_rules = tmpinvalidfile
        self.command_input = tmpinvalidfile
        self.command_prefix = None
        self.command_engine_debug = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        mapping_engine = cli.MappingEngineTester()
        self.assertRaises(SystemExit, mapping_engine.main)

    def test_mapping_engine_tester_with_invalid_input_file(self):
        tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilejsonname = tempfilejson.file_name
        updated_mapping = copy.deepcopy(mapping_fixtures.MAPPING_SMALL)
        with open(tmpfilejsonname, 'w') as f:
            f.write(jsonutils.dumps(updated_mapping))
        self.command_rules = tmpfilejsonname
        self.command_input = 'invalid.csv'
        self.command_prefix = None
        self.command_engine_debug = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        mapping_engine = cli.MappingEngineTester()
        self.assertRaises(SystemExit, mapping_engine.main)

    def test_mapping_engine_tester(self):
        tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilejsonname = tempfilejson.file_name
        updated_mapping = copy.deepcopy(mapping_fixtures.MAPPING_SMALL)
        with open(tmpfilejsonname, 'w') as f:
            f.write(jsonutils.dumps(updated_mapping))
        self.command_rules = tmpfilejsonname
        tempfile = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilename = tempfile.file_name
        with open(tmpfilename, 'w') as f:
            f.write('\n')
            f.write('UserName:me\n')
            f.write('orgPersonType:NoContractor\n')
            f.write('LastName:Bo\n')
            f.write('FirstName:Jill\n')
        self.command_input = tmpfilename
        self.command_prefix = None
        self.command_engine_debug = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        mapping_engine = cli.MappingEngineTester()
        with mock.patch('builtins.print') as mock_print:
            mapping_engine.main()
            self.assertEqual(mock_print.call_count, 3)
            call = mock_print.call_args_list[0]
            args, kwargs = call
            self.assertTrue(args[0].startswith('Using Rules:'))
            call = mock_print.call_args_list[1]
            args, kwargs = call
            self.assertTrue(args[0].startswith('Using Assertion:'))
            call = mock_print.call_args_list[2]
            args, kwargs = call
            expected = {'group_names': [], 'user': {'type': 'ephemeral', 'name': 'me'}, 'projects': [], 'group_ids': ['0cd5e9']}
            self.assertEqual(jsonutils.loads(args[0]), expected)

    def test_mapping_engine_tester_with_invalid_data(self):
        tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilejsonname = tempfilejson.file_name
        updated_mapping = copy.deepcopy(mapping_fixtures.MAPPING_SMALL)
        with open(tmpfilejsonname, 'w') as f:
            f.write(jsonutils.dumps(updated_mapping))
        self.command_rules = tmpfilejsonname
        tempfile = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilename = tempfile.file_name
        with open(tmpfilename, 'w') as f:
            f.write('\n')
            f.write('UserName: me\n')
            f.write('Email: No@example.com\n')
        self.command_input = tmpfilename
        self.command_prefix = None
        self.command_engine_debug = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        mapping_engine = cli.MappingEngineTester()
        self.assertRaises(exception.ValidationError, mapping_engine.main)

    def test_mapping_engine_tester_logs_direct_maps(self):
        tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilejsonname = tempfilejson.file_name
        updated_mapping = copy.deepcopy(mapping_fixtures.MAPPING_SMALL)
        with open(tmpfilejsonname, 'w') as f:
            f.write(jsonutils.dumps(updated_mapping))
        self.command_rules = tmpfilejsonname
        tempfile = self.useFixture(temporaryfile.SecureTempFile())
        tmpfilename = tempfile.file_name
        with open(tmpfilename, 'w') as f:
            f.write('\n')
            f.write('UserName:me\n')
            f.write('orgPersonType:NoContractor\n')
            f.write('LastName:Bo\n')
            f.write('FirstName:Jill\n')
        self.command_input = tmpfilename
        self.command_prefix = None
        self.command_engine_debug = True
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        mapping_engine = cli.MappingEngineTester()
        logging = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
        mapping_engine.main()
        expected_msg = "direct_maps: [['me']]"
        self.assertThat(logging.output, matchers.Contains(expected_msg))