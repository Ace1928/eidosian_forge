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
class TestDomainConfigFinder(unit.BaseTestCase):

    def setUp(self):
        super(TestDomainConfigFinder, self).setUp()
        self.logging = self.useFixture(fixtures.LoggerFixture())

    @mock.patch('os.walk')
    def test_finder_ignores_files(self, mock_walk):
        mock_walk.return_value = [['.', [], ['file.txt', 'keystone.conf', 'keystone.domain0.conf']]]
        domain_configs = list(cli._domain_config_finder('.'))
        expected_domain_configs = [('./keystone.domain0.conf', 'domain0')]
        self.assertThat(domain_configs, matchers.Equals(expected_domain_configs))
        expected_msg_template = 'Ignoring file (%s) while scanning domain config directory'
        self.assertThat(self.logging.output, matchers.Contains(expected_msg_template % 'file.txt'))
        self.assertThat(self.logging.output, matchers.Contains(expected_msg_template % 'keystone.conf'))