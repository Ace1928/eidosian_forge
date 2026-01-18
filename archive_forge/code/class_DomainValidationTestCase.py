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
class DomainValidationTestCase(unit.BaseTestCase):
    """Test for V3 Domain API validation."""

    def setUp(self):
        super(DomainValidationTestCase, self).setUp()
        self.domain_name = 'My Domain'
        create = resource_schema.domain_create
        update = resource_schema.domain_update
        self.create_domain_validator = validators.SchemaValidator(create)
        self.update_domain_validator = validators.SchemaValidator(update)

    def test_validate_domain_request(self):
        """Make sure we successfully validate a create domain request."""
        request_to_validate = {'name': self.domain_name}
        self.create_domain_validator.validate(request_to_validate)

    def test_validate_domain_request_without_name_fails(self):
        """Make sure we raise an exception when `name` isn't included."""
        request_to_validate = {'enabled': True}
        self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_request_with_enabled(self):
        """Validate `enabled` as boolean-like values for domains."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.domain_name, 'enabled': valid_enabled}
            self.create_domain_validator.validate(request_to_validate)

    def test_validate_domain_request_with_invalid_enabled_fails(self):
        """Exception is raised when `enabled` isn't a boolean-like value."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'name': self.domain_name, 'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_request_with_valid_description(self):
        """Test that we validate `description` in create domain requests."""
        request_to_validate = {'name': self.domain_name, 'description': 'My Domain'}
        self.create_domain_validator.validate(request_to_validate)

    def test_validate_domain_request_with_invalid_description_fails(self):
        """Exception is raised when `description` is a non-string value."""
        request_to_validate = {'name': self.domain_name, 'description': False}
        self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_request_with_name_too_long(self):
        """Exception is raised when `name` is too long."""
        long_domain_name = 'a' * 65
        request_to_validate = {'name': long_domain_name}
        self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_create_fails_with_invalid_name(self):
        """Exception when validating a create request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_create_with_tags(self):
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', 'bar']}
        self.create_domain_validator.validate(request_to_validate)

    def test_validate_domain_create_with_tags_invalid_char(self):
        invalid_chars = [',', '/']
        for char in invalid_chars:
            tag = uuid.uuid4().hex + char
            request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', tag]}
            self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_create_with_tag_name_too_long(self):
        invalid_name = 'a' * 256
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', invalid_name]}
        self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_create_with_too_many_tags(self):
        tags = [uuid.uuid4().hex for _ in range(81)]
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': tags}
        self.assertRaises(exception.SchemaValidationError, self.create_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_request(self):
        """Test that we validate a domain update request."""
        request_to_validate = {'domain_id': uuid.uuid4().hex}
        self.update_domain_validator.validate(request_to_validate)

    def test_validate_domain_update_request_with_no_parameters_fails(self):
        """Exception is raised when updating a domain without parameters."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_request_with_name_too_long_fails(self):
        """Exception raised when updating a domain with `name` too long."""
        long_domain_name = 'a' * 65
        request_to_validate = {'name': long_domain_name}
        self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_fails_with_invalid_name(self):
        """Exception when validating an update request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_with_tags(self):
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', 'bar']}
        self.update_domain_validator.validate(request_to_validate)

    def test_validate_domain_update_with_tags_invalid_char(self):
        invalid_chars = [',', '/']
        for char in invalid_chars:
            tag = uuid.uuid4().hex + char
            request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', tag]}
            self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_with_tag_name_too_long(self):
        invalid_name = 'a' * 256
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': ['foo', invalid_name]}
        self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)

    def test_validate_domain_update_with_too_many_tags(self):
        tags = [uuid.uuid4().hex for _ in range(81)]
        request_to_validate = {'name': uuid.uuid4().hex, 'tags': tags}
        self.assertRaises(exception.SchemaValidationError, self.update_domain_validator.validate, request_to_validate)