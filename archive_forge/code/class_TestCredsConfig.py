from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from gslib.cloud_api import AccessDeniedException
from gslib.cred_types import CredTypes
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
from datetime import datetime
class TestCredsConfig(testcase.GsUtilUnitTestCase):
    """Tests for various combinations of configured credentials."""

    def setUp(self):
        super(TestCredsConfig, self).setUp()
        self.log_handler = MockLoggingHandler()
        self.logger.addHandler(self.log_handler)

    def testMultipleConfiguredCreds(self):
        with SetBotoConfigForTest([('Credentials', 'gs_oauth2_refresh_token', 'foo'), ('Credentials', 'gs_service_client_id', 'bar'), ('Credentials', 'gs_service_key_file', 'baz'), ('Credentials', 'gs_impersonate_service_account', None)]):
            try:
                GcsJsonApi(None, self.logger, DiscardMessagesQueue())
                self.fail('Succeeded with multiple types of configured creds.')
            except CommandException as e:
                msg = str(e)
                self.assertIn('types of configured credentials', msg)
                self.assertIn(CredTypes.OAUTH2_USER_ACCOUNT, msg)
                self.assertIn(CredTypes.OAUTH2_SERVICE_ACCOUNT, msg)

    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    @mock.patch('gslib.third_party.iamcredentials_apitools.iamcredentials_v1_client.IamcredentialsV1.ProjectsServiceAccountsService.GenerateAccessToken')
    def testImpersonationBlockedByIamCredentialsApiErrors(self, mock_iam_creds_generate_access_token):
        with SetBotoConfigForTest([('Credentials', 'gs_oauth2_refresh_token', 'foo'), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_impersonate_service_account', 'bar')]):
            mock_iam_creds_generate_access_token.side_effect = _Make403('The caller does not have permission')
            try:
                GcsJsonApi(None, self.logger, DiscardMessagesQueue())
                self.fail('Succeeded when IAM Credentials threw an error')
            except AccessDeniedException as e:
                self.assertIn('Service account impersonation failed.', str(e))
            mock_iam_creds_generate_access_token.side_effect = _Make403('IAM Service Account Credentials API has not been used')
            try:
                GcsJsonApi(None, self.logger, DiscardMessagesQueue())
                self.fail('Succeeded when IAM Credentials threw an error')
            except AccessDeniedException as e:
                self.assertIn('IAM Service Account Credentials API has not', str(e))

    @SkipForS3('Tests only uses gs credentials.')
    @SkipForXML('Tests only run on JSON API.')
    @mock.patch('gslib.third_party.iamcredentials_apitools.iamcredentials_v1_client.IamcredentialsV1.ProjectsServiceAccountsService.GenerateAccessToken')
    def testImpersonationSuccessfullyUsesToken(self, mock_iam_creds_generate_access_token):
        with SetBotoConfigForTest([('Credentials', 'gs_oauth2_refresh_token', 'foo'), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_impersonate_service_account', 'bar')]):
            fake_token = 'Mock token from IAM Credentials API'
            expire_time = datetime.now().strftime('%Y-%m-%dT23:59:59Z')
            mock_iam_creds_generate_access_token.return_value.accessToken = fake_token
            mock_iam_creds_generate_access_token.return_value.expireTime = expire_time
            api = GcsJsonApi(None, self.logger, DiscardMessagesQueue())
            self.assertIn(fake_token, str(api.credentials.access_token))