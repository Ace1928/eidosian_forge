import boto
from tests.compat import unittest
class CognitoTest(unittest.TestCase):

    def setUp(self):
        self.cognito_identity = boto.connect_cognito_identity()
        self.cognito_sync = boto.connect_cognito_sync()
        self.identity_pool_name = 'myIdentityPool'
        response = self.cognito_identity.create_identity_pool(identity_pool_name=self.identity_pool_name, allow_unauthenticated_identities=False)
        self.identity_pool_id = response['IdentityPoolId']

    def tearDown(self):
        self.cognito_identity.delete_identity_pool(identity_pool_id=self.identity_pool_id)