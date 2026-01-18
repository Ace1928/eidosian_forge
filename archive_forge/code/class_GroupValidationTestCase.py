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
class GroupValidationTestCase(unit.BaseTestCase):
    """Test for V3 Group API validation."""

    def setUp(self):
        super(GroupValidationTestCase, self).setUp()
        self.group_name = uuid.uuid4().hex
        create = identity_schema.group_create
        update = identity_schema.group_update
        self.create_group_validator = validators.SchemaValidator(create)
        self.update_group_validator = validators.SchemaValidator(update)

    def test_validate_group_create_succeeds(self):
        """Validate create group requests."""
        request_to_validate = {'name': self.group_name}
        self.create_group_validator.validate(request_to_validate)

    def test_validate_group_create_succeeds_with_all_parameters(self):
        """Validate create group requests with all parameters."""
        request_to_validate = {'name': self.group_name, 'description': uuid.uuid4().hex, 'domain_id': uuid.uuid4().hex}
        self.create_group_validator.validate(request_to_validate)

    def test_validate_group_create_fails_without_group_name(self):
        """Exception raised when group name is not provided in request."""
        request_to_validate = {'description': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_group_validator.validate, request_to_validate)

    def test_validate_group_create_succeeds_with_extra_parameters(self):
        """Validate extra attributes on group create requests."""
        request_to_validate = {'name': self.group_name, 'other_attr': uuid.uuid4().hex}
        self.create_group_validator.validate(request_to_validate)

    def test_validate_group_create_fails_with_invalid_name(self):
        """Exception when validating a create request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.create_group_validator.validate, request_to_validate)

    def test_validate_group_update_succeeds(self):
        """Validate group update requests."""
        request_to_validate = {'description': uuid.uuid4().hex}
        self.update_group_validator.validate(request_to_validate)

    def test_validate_group_update_fails_with_no_parameters(self):
        """Exception raised when no parameters passed in on update."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_group_validator.validate, request_to_validate)

    def test_validate_group_update_succeeds_with_extra_parameters(self):
        """Validate group update requests with extra parameters."""
        request_to_validate = {'other_attr': uuid.uuid4().hex}
        self.update_group_validator.validate(request_to_validate)

    def test_validate_group_update_fails_with_invalid_name(self):
        """Exception when validating an update request with invalid `name`."""
        for invalid_name in _INVALID_NAMES:
            request_to_validate = {'name': invalid_name}
            self.assertRaises(exception.SchemaValidationError, self.update_group_validator.validate, request_to_validate)