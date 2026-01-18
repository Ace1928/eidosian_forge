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
class RoleValidationTestCase(unit.BaseTestCase):
    """Test for V3 Role API validation."""

    def setUp(self):
        super(RoleValidationTestCase, self).setUp()
        self.role_name = 'My Role'
        create = assignment_schema.role_create
        update = assignment_schema.role_update
        self.create_role_validator = validators.SchemaValidator(create)
        self.update_role_validator = validators.SchemaValidator(update)

    def test_validate_role_request(self):
        """Test we can successfully validate a create role request."""
        request_to_validate = {'name': self.role_name}
        self.create_role_validator.validate(request_to_validate)

    def test_validate_role_create_without_name_raises_exception(self):
        """Test that we raise an exception when `name` isn't included."""
        request_to_validate = {'enabled': True}
        self.assertRaises(exception.SchemaValidationError, self.create_role_validator.validate, request_to_validate)

    def test_validate_role_create_fails_with_invalid_name(self):
        """Exception when validating a create request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.create_role_validator.validate, request_to_validate)

    def test_validate_role_create_request_with_name_too_long_fails(self):
        """Exception raised when creating a role with `name` too long."""
        long_role_name = 'a' * 256
        request_to_validate = {'name': long_role_name}
        self.assertRaises(exception.SchemaValidationError, self.create_role_validator.validate, request_to_validate)

    def test_validate_role_request_with_valid_description(self):
        """Test we can validate`description` in create role request."""
        request_to_validate = {'name': self.role_name, 'description': 'My Role'}
        self.create_role_validator.validate(request_to_validate)

    def test_validate_role_request_fails_with_invalid_description(self):
        """Exception is raised when `description` as a non-string value."""
        request_to_validate = {'name': self.role_name, 'description': False}
        self.assertRaises(exception.SchemaValidationError, self.create_role_validator.validate, request_to_validate)

    def test_validate_role_update_request(self):
        """Test that we validate a role update request."""
        request_to_validate = {'name': 'My New Role'}
        self.update_role_validator.validate(request_to_validate)

    def test_validate_role_update_fails_with_invalid_name(self):
        """Exception when validating an update request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.update_role_validator.validate, request_to_validate)

    def test_validate_role_update_request_with_name_too_long_fails(self):
        """Exception raised when updating a role with `name` too long."""
        long_role_name = 'a' * 256
        request_to_validate = {'name': long_role_name}
        self.assertRaises(exception.SchemaValidationError, self.update_role_validator.validate, request_to_validate)