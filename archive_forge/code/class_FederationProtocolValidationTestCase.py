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
class FederationProtocolValidationTestCase(unit.BaseTestCase):
    """Test for V3 Federation Protocol API validation."""

    def setUp(self):
        super(FederationProtocolValidationTestCase, self).setUp()
        create = federation_schema.protocol_create
        update = federation_schema.protocol_update
        self.create_protocol_validator = validators.SchemaValidator(create)
        self.update_protocol_validator = validators.SchemaValidator(update)

    def test_validate_protocol_request_succeeds(self):
        """Test that we validate a protocol request successfully."""
        request_to_validate = {'mapping_id': uuid.uuid4().hex}
        self.create_protocol_validator.validate(request_to_validate)

    def test_validate_protocol_request_succeeds_with_nonuuid_mapping_id(self):
        """Test that we allow underscore in mapping_id value."""
        request_to_validate = {'mapping_id': 'my_mapping_id'}
        self.create_protocol_validator.validate(request_to_validate)

    def test_validate_protocol_request_fails_with_invalid_params(self):
        """Exception raised when unknown parameter is found."""
        request_to_validate = {'bogus': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_protocol_validator.validate, request_to_validate)

    def test_validate_protocol_request_no_parameters(self):
        """Test that schema validation with empty request body."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.create_protocol_validator.validate, request_to_validate)

    def test_validate_protocol_request_fails_with_invalid_mapping_id(self):
        """Exception raised when mapping_id is not string."""
        request_to_validate = {'mapping_id': 12334}
        self.assertRaises(exception.SchemaValidationError, self.create_protocol_validator.validate, request_to_validate)

    def test_validate_protocol_request_succeeds_on_update(self):
        """Test that we validate a protocol update request successfully."""
        request_to_validate = {'mapping_id': uuid.uuid4().hex}
        self.update_protocol_validator.validate(request_to_validate)

    def test_validate_update_protocol_request_succeeds_with_nonuuid_id(self):
        """Test that we allow underscore in mapping_id value when updating."""
        request_to_validate = {'mapping_id': 'my_mapping_id'}
        self.update_protocol_validator.validate(request_to_validate)

    def test_validate_update_protocol_request_fails_with_invalid_params(self):
        """Exception raised when unknown parameter in protocol update."""
        request_to_validate = {'bogus': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.update_protocol_validator.validate, request_to_validate)

    def test_validate_update_protocol_with_no_parameters_fails(self):
        """Test that updating a protocol requires at least one attribute."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_protocol_validator.validate, request_to_validate)

    def test_validate_update_protocol_request_fails_with_invalid_id(self):
        """Test that updating a protocol with a non-string mapping_id fail."""
        for bad_mapping_id in [12345, True]:
            request_to_validate = {'mapping_id': bad_mapping_id}
            self.assertRaises(exception.SchemaValidationError, self.update_protocol_validator.validate, request_to_validate)