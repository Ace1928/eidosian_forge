import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestApplicationCredentialShow(TestApplicationCredential):

    def setUp(self):
        super(TestApplicationCredentialShow, self).setUp()
        self.app_creds_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.APP_CRED_BASIC), loaded=True)
        self.cmd = application_credential.ShowApplicationCredential(self.app, None)

    def test_application_credential_show(self):
        arglist = [identity_fakes.app_cred_id]
        verifylist = [('application_credential', identity_fakes.app_cred_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.app_creds_mock.get.assert_called_with(identity_fakes.app_cred_id)
        collist = ('access_rules', 'description', 'expires_at', 'id', 'name', 'project_id', 'roles', 'secret', 'unrestricted')
        self.assertEqual(collist, columns)
        datalist = (None, None, None, identity_fakes.app_cred_id, identity_fakes.app_cred_name, identity_fakes.project_id, identity_fakes.role_name, identity_fakes.app_cred_secret, False)
        self.assertEqual(datalist, data)