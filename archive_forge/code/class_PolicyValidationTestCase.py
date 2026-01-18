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
class PolicyValidationTestCase(unit.BaseTestCase):
    """Test for V3 Policy API validation."""

    def setUp(self):
        super(PolicyValidationTestCase, self).setUp()
        create = policy_schema.policy_create
        update = policy_schema.policy_update
        self.create_policy_validator = validators.SchemaValidator(create)
        self.update_policy_validator = validators.SchemaValidator(update)

    def test_validate_policy_succeeds(self):
        """Test that we validate a create policy request."""
        request_to_validate = {'blob': 'some blob information', 'type': 'application/json'}
        self.create_policy_validator.validate(request_to_validate)

    def test_validate_policy_without_blob_fails(self):
        """Exception raised without `blob` in request."""
        request_to_validate = {'type': 'application/json'}
        self.assertRaises(exception.SchemaValidationError, self.create_policy_validator.validate, request_to_validate)

    def test_validate_policy_without_type_fails(self):
        """Exception raised without `type` in request."""
        request_to_validate = {'blob': 'some blob information'}
        self.assertRaises(exception.SchemaValidationError, self.create_policy_validator.validate, request_to_validate)

    def test_validate_policy_create_with_extra_parameters_succeeds(self):
        """Validate policy create with extra parameters."""
        request_to_validate = {'blob': 'some blob information', 'type': 'application/json', 'extra': 'some extra stuff'}
        self.create_policy_validator.validate(request_to_validate)

    def test_validate_policy_create_with_invalid_type_fails(self):
        """Exception raised when `blob` and `type` are boolean."""
        for prop in ['blob', 'type']:
            request_to_validate = {prop: False}
            self.assertRaises(exception.SchemaValidationError, self.create_policy_validator.validate, request_to_validate)

    def test_validate_policy_update_without_parameters_fails(self):
        """Exception raised when updating policy without parameters."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_policy_validator.validate, request_to_validate)

    def test_validate_policy_update_with_extra_parameters_succeeds(self):
        """Validate policy update request with extra parameters."""
        request_to_validate = {'blob': 'some blob information', 'type': 'application/json', 'extra': 'some extra stuff'}
        self.update_policy_validator.validate(request_to_validate)

    def test_validate_policy_update_succeeds(self):
        """Test that we validate a policy update request."""
        request_to_validate = {'blob': 'some blob information', 'type': 'application/json'}
        self.update_policy_validator.validate(request_to_validate)

    def test_validate_policy_update_with_invalid_type_fails(self):
        """Exception raised when invalid `type` on policy update."""
        for prop in ['blob', 'type']:
            request_to_validate = {prop: False}
            self.assertRaises(exception.SchemaValidationError, self.update_policy_validator.validate, request_to_validate)