from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def _expect_failure(self, post_data):
    self.assertRaises(exception.SchemaValidationError, schema.validate_issue_token_auth, post_data)