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
class RegionValidationTestCase(unit.BaseTestCase):
    """Test for V3 Region API validation."""

    def setUp(self):
        super(RegionValidationTestCase, self).setUp()
        self.region_name = 'My Region'
        create = catalog_schema.region_create
        update = catalog_schema.region_update
        self.create_region_validator = validators.SchemaValidator(create)
        self.update_region_validator = validators.SchemaValidator(update)

    def test_validate_region_request(self):
        """Test that we validate a basic region request."""
        request_to_validate = {}
        self.create_region_validator.validate(request_to_validate)

    def test_validate_region_create_request_with_parameters(self):
        """Test that we validate a region request with parameters."""
        request_to_validate = {'id': 'us-east', 'description': 'US East Region', 'parent_region_id': 'US Region'}
        self.create_region_validator.validate(request_to_validate)

    def test_validate_region_create_with_uuid(self):
        """Test that we validate a region request with a UUID as the id."""
        request_to_validate = {'id': uuid.uuid4().hex, 'description': 'US East Region', 'parent_region_id': uuid.uuid4().hex}
        self.create_region_validator.validate(request_to_validate)

    def test_validate_region_create_fails_with_invalid_region_id(self):
        """Exception raised when passing invalid `id` in request."""
        request_to_validate = {'id': 1234, 'description': 'US East Region'}
        self.assertRaises(exception.SchemaValidationError, self.create_region_validator.validate, request_to_validate)

    def test_validate_region_create_succeeds_with_extra_parameters(self):
        """Validate create region request with extra values."""
        request_to_validate = {'other_attr': uuid.uuid4().hex}
        self.create_region_validator.validate(request_to_validate)

    def test_validate_region_create_succeeds_with_no_parameters(self):
        """Validate create region request with no parameters."""
        request_to_validate = {}
        self.create_region_validator.validate(request_to_validate)

    def test_validate_region_update_succeeds(self):
        """Test that we validate a region update request."""
        request_to_validate = {'id': 'us-west', 'description': 'US West Region', 'parent_region_id': 'us-region'}
        self.update_region_validator.validate(request_to_validate)

    def test_validate_region_update_succeeds_with_extra_parameters(self):
        """Validate extra attributes in the region update request."""
        request_to_validate = {'other_attr': uuid.uuid4().hex}
        self.update_region_validator.validate(request_to_validate)

    def test_validate_region_update_fails_with_no_parameters(self):
        """Exception raised when passing no parameters in a region update."""
        request_to_validate = {}
        self.assertRaises(exception.SchemaValidationError, self.update_region_validator.validate, request_to_validate)