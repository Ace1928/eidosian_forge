import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class PolicyFileTestCase(unit.TestCase):

    def setUp(self):
        self.tempfile = self.useFixture(temporaryfile.SecureTempFile())
        self.tmpfilename = self.tempfile.file_name
        super(PolicyFileTestCase, self).setUp()
        self.target = {}

    def _policy_fixture(self):
        return ksfixtures.Policy(self.config_fixture, policy_file=self.tmpfilename)

    def test_modified_policy_reloads(self):
        action = 'example:test'
        empty_credentials = {}
        with open(self.tmpfilename, 'w') as policyfile:
            policyfile.write('{"example:test": []}')
        policy.enforce(empty_credentials, action, self.target)
        with open(self.tmpfilename, 'w') as policyfile:
            policyfile.write('{"example:test": ["false:false"]}')
        policy._ENFORCER._enforcer.clear()
        self.assertRaises(exception.ForbiddenAction, policy.enforce, empty_credentials, action, self.target)