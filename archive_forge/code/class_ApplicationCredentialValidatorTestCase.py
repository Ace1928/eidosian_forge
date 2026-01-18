import copy
import uuid
from keystone.application_credential import schema as app_cred_schema
from keystone.assignment import schema as assignment_schema
from keystone.catalog import schema as catalog_schema
from keystone.common import validation
from keystone.common.validation import parameter_types
from keystone.common.validation import validators
from keystone.credential import schema as credential_schema
from keystone import exception
from keystone.federation import schema as federation_schema
from keystone.identity.backends import resource_options as ro
from keystone.identity import schema as identity_schema
from keystone.limit import schema as limit_schema
from keystone.oauth1 import schema as oauth1_schema
from keystone.policy import schema as policy_schema
from keystone.resource import schema as resource_schema
from keystone.tests import unit
from keystone.trust import schema as trust_schema
class ApplicationCredentialValidatorTestCase(unit.TestCase):
    _valid_roles = [{'name': 'member'}, {'id': uuid.uuid4().hex}, {'id': str(uuid.uuid4())}, {'name': '_member_'}]
    _invalid_roles = [True, 123, None, {'badkey': 'badval'}]

    def setUp(self):
        super(ApplicationCredentialValidatorTestCase, self).setUp()
        create = app_cred_schema.application_credential_create
        self.create_app_cred_validator = validators.SchemaValidator(create)

    def test_validate_app_cred_request(self):
        request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [{'name': 'member'}], 'expires_at': 'tomorrow'}
        self.create_app_cred_validator.validate(request_to_validate)

    def test_validate_app_cred_request_without_name_fails(self):
        request_to_validate = {'description': 'My App Cred', 'roles': [{'name': 'member'}], 'expires_at': 'tomorrow'}
        self.assertRaises(exception.SchemaValidationError, self.create_app_cred_validator.validate, request_to_validate)

    def test_validate_app_cred_with_invalid_expires_at_fails(self):
        request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [{'name': 'member'}], 'expires_at': 3}
        self.assertRaises(exception.SchemaValidationError, self.create_app_cred_validator.validate, request_to_validate)

    def test_validate_app_cred_with_null_expires_at_succeeds(self):
        request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [{'name': 'member'}]}
        self.create_app_cred_validator.validate(request_to_validate)

    def test_validate_app_cred_with_unrestricted_flag_succeeds(self):
        request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [{'name': 'member'}], 'unrestricted': True}
        self.create_app_cred_validator.validate(request_to_validate)

    def test_validate_app_cred_with_secret_succeeds(self):
        request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [{'name': 'member'}], 'secret': 'secretsecretsecretsecret'}
        self.create_app_cred_validator.validate(request_to_validate)

    def test_validate_app_cred_invalid_roles_fails(self):
        for role in self._invalid_roles:
            request_to_validate = {'name': 'myappcred', 'description': 'My App Cred', 'roles': [role]}
            self.assertRaises(exception.SchemaValidationError, self.create_app_cred_validator.validate, request_to_validate)