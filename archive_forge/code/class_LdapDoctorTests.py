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
class LdapDoctorTests(unit.TestCase):

    def test_user_enabled_emulation_dn_ignored_raised(self):
        self.config_fixture.config(group='ldap', user_enabled_emulation=False)
        self.config_fixture.config(group='ldap', user_enabled_emulation_dn='cn=enabled_users,dc=example,dc=com')
        self.assertTrue(ldap.symptom_LDAP_user_enabled_emulation_dn_ignored())

    def test_user_enabled_emulation_dn_ignored_not_raised(self):
        self.config_fixture.config(group='ldap', user_enabled_emulation=True)
        self.config_fixture.config(group='ldap', user_enabled_emulation_dn='cn=enabled_users,dc=example,dc=com')
        self.assertFalse(ldap.symptom_LDAP_user_enabled_emulation_dn_ignored())
        self.config_fixture.config(group='ldap', user_enabled_emulation=False)
        self.config_fixture.config(group='ldap', user_enabled_emulation_dn=None)
        self.assertFalse(ldap.symptom_LDAP_user_enabled_emulation_dn_ignored())

    def test_user_enabled_emulation_use_group_config_ignored_raised(self):
        self.config_fixture.config(group='ldap', user_enabled_emulation=False)
        self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=True)
        self.assertTrue(ldap.symptom_LDAP_user_enabled_emulation_use_group_config_ignored())

    def test_user_enabled_emulation_use_group_config_ignored_not_raised(self):
        self.config_fixture.config(group='ldap', user_enabled_emulation=False)
        self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=False)
        self.assertFalse(ldap.symptom_LDAP_user_enabled_emulation_use_group_config_ignored())
        self.config_fixture.config(group='ldap', user_enabled_emulation=True)
        self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=True)
        self.assertFalse(ldap.symptom_LDAP_user_enabled_emulation_use_group_config_ignored())

    def test_group_members_are_ids_disabled_raised(self):
        self.config_fixture.config(group='ldap', group_objectclass='posixGroup')
        self.config_fixture.config(group='ldap', group_members_are_ids=False)
        self.assertTrue(ldap.symptom_LDAP_group_members_are_ids_disabled())

    def test_group_members_are_ids_disabled_not_raised(self):
        self.config_fixture.config(group='ldap', group_objectclass='posixGroup')
        self.config_fixture.config(group='ldap', group_members_are_ids=True)
        self.assertFalse(ldap.symptom_LDAP_group_members_are_ids_disabled())
        self.config_fixture.config(group='ldap', group_objectclass='groupOfNames')
        self.config_fixture.config(group='ldap', group_members_are_ids=False)
        self.assertFalse(ldap.symptom_LDAP_group_members_are_ids_disabled())

    @mock.patch('os.listdir')
    @mock.patch('os.path.isdir')
    def test_file_based_domain_specific_configs_raised(self, mocked_isdir, mocked_listdir):
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        mocked_isdir.return_value = False
        self.assertTrue(ldap.symptom_LDAP_file_based_domain_specific_configs())
        mocked_isdir.return_value = True
        mocked_listdir.return_value = ['openstack.domains.conf']
        self.assertTrue(ldap.symptom_LDAP_file_based_domain_specific_configs())

    @mock.patch('os.listdir')
    @mock.patch('os.path.isdir')
    def test_file_based_domain_specific_configs_not_raised(self, mocked_isdir, mocked_listdir):
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=False)
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        self.assertFalse(ldap.symptom_LDAP_file_based_domain_specific_configs())
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        mocked_isdir.return_value = True
        mocked_listdir.return_value = ['keystone.domains.conf']
        self.assertFalse(ldap.symptom_LDAP_file_based_domain_specific_configs())

    @mock.patch('os.listdir')
    @mock.patch('os.path.isdir')
    @mock.patch('keystone.cmd.doctor.ldap.configparser.ConfigParser')
    def test_file_based_domain_specific_configs_formatted_correctly_raised(self, mocked_parser, mocked_isdir, mocked_listdir):
        symptom = 'symptom_LDAP_file_based_domain_specific_configs_formatted_correctly'
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        mocked_isdir.return_value = True
        mocked_listdir.return_value = ['keystone.domains.conf']
        mock_instance = mock.MagicMock()
        mock_instance.read.side_effect = configparser.Error('No Section')
        mocked_parser.return_value = mock_instance
        self.assertTrue(getattr(ldap, symptom)())

    @mock.patch('os.listdir')
    @mock.patch('os.path.isdir')
    def test_file_based_domain_specific_configs_formatted_correctly_not_raised(self, mocked_isdir, mocked_listdir):
        symptom = 'symptom_LDAP_file_based_domain_specific_configs_formatted_correctly'
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=False)
        self.assertFalse(getattr(ldap, symptom)())
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
        self.assertFalse(getattr(ldap, symptom)())
        self.config_fixture.config(group='identity', domain_configurations_from_database=True)
        self.assertFalse(getattr(ldap, symptom)())
        mocked_isdir.return_value = False
        self.assertFalse(getattr(ldap, symptom)())
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        mocked_isdir.return_value = True
        self.assertFalse(getattr(ldap, symptom)())
        mocked_listdir.return_value = ['keystone.domains.conf']
        self.assertFalse(getattr(ldap, symptom)())