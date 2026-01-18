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
class EndpointGroupValidationTestCase(unit.BaseTestCase):
    """Test for V3 Endpoint Group API validation."""

    def setUp(self):
        super(EndpointGroupValidationTestCase, self).setUp()
        create = catalog_schema.endpoint_group_create
        update = catalog_schema.endpoint_group_update
        self.create_endpoint_grp_validator = validators.SchemaValidator(create)
        self.update_endpoint_grp_validator = validators.SchemaValidator(update)

    def test_validate_endpoint_group_request_succeeds(self):
        """Test that we validate an endpoint group request."""
        request_to_validate = {'description': 'endpoint group description', 'filters': {'interface': 'admin'}, 'name': 'endpoint_group_name'}
        self.create_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_endpoint_group_create_succeeds_with_req_parameters(self):
        """Validate required endpoint group parameters.

        This test ensure that validation succeeds with only the required
        parameters passed for creating an endpoint group.
        """
        request_to_validate = {'filters': {'interface': 'admin'}, 'name': 'endpoint_group_name'}
        self.create_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_endpoint_group_create_succeeds_with_valid_filters(self):
        """Validate `filters` in endpoint group create requests."""
        request_to_validate = {'description': 'endpoint group description', 'name': 'endpoint_group_name'}
        for valid_filters in _VALID_FILTERS:
            request_to_validate['filters'] = valid_filters
            self.create_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_create_endpoint_group_fails_with_invalid_filters(self):
        """Validate invalid `filters` value in endpoint group parameters.

        This test ensures that exception is raised when non-dict values is
        used as `filters` in endpoint group create request.
        """
        request_to_validate = {'description': 'endpoint group description', 'name': 'endpoint_group_name'}
        for invalid_filters in _INVALID_FILTERS:
            request_to_validate['filters'] = invalid_filters
            self.assertRaises(exception.SchemaValidationError, self.create_endpoint_grp_validator.validate, request_to_validate)

    def test_validate_endpoint_group_create_fails_without_name(self):
        """Exception raised when `name` isn't in endpoint group request."""
        request_to_validate = {'description': 'endpoint group description', 'filters': {'interface': 'admin'}}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_grp_validator.validate, request_to_validate)

    def test_validate_endpoint_group_create_fails_without_filters(self):
        """Exception raised when `filters` isn't in endpoint group request."""
        request_to_validate = {'description': 'endpoint group description', 'name': 'endpoint_group_name'}
        self.assertRaises(exception.SchemaValidationError, self.create_endpoint_grp_validator.validate, request_to_validate)

    def test_validate_endpoint_group_update_request_succeeds(self):
        """Test that we validate an endpoint group update request."""
        request_to_validate = {'description': 'endpoint group description', 'filters': {'interface': 'admin'}, 'name': 'endpoint_group_name'}
        self.update_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_endpoint_group_update_fails_with_no_parameters(self):
        """Exception raised when no parameters on endpoint group update."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_endpoint_grp_validator.validate, request_to_validate)

    def test_validate_endpoint_group_update_succeeds_with_name(self):
        """Validate request with  only `name` in endpoint group update.

        This test ensures that passing only a `name` passes validation
        on update endpoint group request.
        """
        request_to_validate = {'name': 'endpoint_group_name'}
        self.update_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_endpoint_group_update_succeeds_with_valid_filters(self):
        """Validate `filters` as dict values."""
        for valid_filters in _VALID_FILTERS:
            request_to_validate = {'filters': valid_filters}
            self.update_endpoint_grp_validator.validate(request_to_validate)

    def test_validate_endpoint_group_update_fails_with_invalid_filters(self):
        """Exception raised when passing invalid `filters` in request."""
        for invalid_filters in _INVALID_FILTERS:
            request_to_validate = {'filters': invalid_filters}
            self.assertRaises(exception.SchemaValidationError, self.update_endpoint_grp_validator.validate, request_to_validate)