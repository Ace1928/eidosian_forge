from boto.cognito.sync.exceptions import ResourceNotFoundException
from tests.integration.cognito import CognitoTest
class TestCognitoSync(CognitoTest):
    """
    Even more so for Cognito Sync, Cognito identites are required.  However,
    AWS account IDs are required to aqcuire a Cognito identity so only
    Cognito pool identity related operations are tested.
    """

    def test_cognito_sync(self):
        response = self.cognito_sync.describe_identity_pool_usage(identity_pool_id=self.identity_pool_id)
        identity_pool_usage = response['IdentityPoolUsage']
        self.assertEqual(identity_pool_usage['SyncSessionsCount'], None)
        self.assertEqual(identity_pool_usage['DataStorage'], 0)

    def test_resource_not_found_exception(self):
        with self.assertRaises(ResourceNotFoundException):
            self.cognito_sync.describe_identity_pool_usage(identity_pool_id='us-east-0:c09e640-b014-4822-86b9-ec77c40d8d6f')