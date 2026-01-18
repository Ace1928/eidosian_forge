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
class ServiceValidationTestCase(unit.BaseTestCase):
    """Test for V3 Service API validation."""

    def setUp(self):
        super(ServiceValidationTestCase, self).setUp()
        create = catalog_schema.service_create
        update = catalog_schema.service_update
        self.create_service_validator = validators.SchemaValidator(create)
        self.update_service_validator = validators.SchemaValidator(update)

    def test_validate_service_create_succeeds(self):
        """Test that we validate a service create request."""
        request_to_validate = {'name': 'Nova', 'description': 'OpenStack Compute Service', 'enabled': True, 'type': 'compute'}
        self.create_service_validator.validate(request_to_validate)

    def test_validate_service_create_succeeds_with_required_parameters(self):
        """Validate a service create request with the required parameters."""
        request_to_validate = {'type': 'compute'}
        self.create_service_validator.validate(request_to_validate)

    def test_validate_service_create_fails_without_type(self):
        """Exception raised when trying to create a service without `type`."""
        request_to_validate = {'name': 'Nova'}
        self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_create_succeeds_with_extra_parameters(self):
        """Test that extra parameters pass validation on create service."""
        request_to_validate = {'other_attr': uuid.uuid4().hex, 'type': uuid.uuid4().hex}
        self.create_service_validator.validate(request_to_validate)

    def test_validate_service_create_succeeds_with_valid_enabled(self):
        """Validate boolean values as enabled values on service create."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': valid_enabled, 'type': uuid.uuid4().hex}
            self.create_service_validator.validate(request_to_validate)

    def test_validate_service_create_fails_with_invalid_enabled(self):
        """Exception raised when boolean-like parameters as `enabled`.

        On service create, make sure an exception is raised if `enabled` is
        not a boolean value.
        """
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': invalid_enabled, 'type': uuid.uuid4().hex}
            self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_create_fails_when_name_too_long(self):
        """Exception raised when `name` is greater than 255 characters."""
        long_name = 'a' * 256
        request_to_validate = {'type': 'compute', 'name': long_name}
        self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_create_fails_when_name_too_short(self):
        """Exception is raised when `name` is too short."""
        request_to_validate = {'type': 'compute', 'name': ''}
        self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_create_fails_when_type_too_long(self):
        """Exception is raised when `type` is too long."""
        long_type_name = 'a' * 256
        request_to_validate = {'type': long_type_name}
        self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_create_fails_when_type_too_short(self):
        """Exception is raised when `type` is too short."""
        request_to_validate = {'type': ''}
        self.assertRaises(exception.SchemaValidationError, self.create_service_validator.validate, request_to_validate)

    def test_validate_service_update_request_succeeds(self):
        """Test that we validate a service update request."""
        request_to_validate = {'name': 'Cinder', 'type': 'volume', 'description': 'OpenStack Block Storage', 'enabled': False}
        self.update_service_validator.validate(request_to_validate)

    def test_validate_service_update_fails_with_no_parameters(self):
        """Exception raised when updating a service without values."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)

    def test_validate_service_update_succeeds_with_extra_parameters(self):
        """Validate updating a service with extra parameters."""
        request_to_validate = {'other_attr': uuid.uuid4().hex}
        self.update_service_validator.validate(request_to_validate)

    def test_validate_service_update_succeeds_with_valid_enabled(self):
        """Validate boolean formats as `enabled` on service update."""
        for valid_enabled in _VALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': valid_enabled}
            self.update_service_validator.validate(request_to_validate)

    def test_validate_service_update_fails_with_invalid_enabled(self):
        """Exception raised when boolean-like values as `enabled`."""
        for invalid_enabled in _INVALID_ENABLED_FORMATS:
            request_to_validate = {'enabled': invalid_enabled}
            self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)

    def test_validate_service_update_fails_with_name_too_long(self):
        """Exception is raised when `name` is too long on update."""
        long_name = 'a' * 256
        request_to_validate = {'name': long_name}
        self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)

    def test_validate_service_update_fails_with_name_too_short(self):
        """Exception is raised when `name` is too short on update."""
        request_to_validate = {'name': ''}
        self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)

    def test_validate_service_update_fails_with_type_too_long(self):
        """Exception is raised when `type` is too long on update."""
        long_type_name = 'a' * 256
        request_to_validate = {'type': long_type_name}
        self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)

    def test_validate_service_update_fails_with_type_too_short(self):
        """Exception is raised when `type` is too short on update."""
        request_to_validate = {'type': ''}
        self.assertRaises(exception.SchemaValidationError, self.update_service_validator.validate, request_to_validate)