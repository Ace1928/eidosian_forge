import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import trust
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestTrustCreate(TestTrust):

    def setUp(self):
        super(TestTrustCreate, self).setUp()
        self.projects_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROJECT), loaded=True)
        self.users_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.USER), loaded=True)
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE), loaded=True)
        self.trusts_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.TRUST), loaded=True)
        self.cmd = trust.CreateTrust(self.app, None)

    def test_trust_create_basic(self):
        arglist = ['--project', identity_fakes.project_id, '--role', identity_fakes.role_id, identity_fakes.user_id, identity_fakes.user_id]
        verifylist = [('project', identity_fakes.project_id), ('impersonate', False), ('role', [identity_fakes.role_id]), ('trustor', identity_fakes.user_id), ('trustee', identity_fakes.user_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'impersonation': False, 'project': identity_fakes.project_id, 'role_ids': [identity_fakes.role_id], 'expires_at': None}
        self.trusts_mock.create.assert_called_with(identity_fakes.user_id, identity_fakes.user_id, **kwargs)
        collist = ('expires_at', 'id', 'impersonation', 'project_id', 'roles', 'trustee_user_id', 'trustor_user_id')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.trust_expires, identity_fakes.trust_id, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.role_name, identity_fakes.user_id, identity_fakes.user_id)
        self.assertEqual(datalist, data)