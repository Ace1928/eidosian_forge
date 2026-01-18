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
class LimitValidationTestCase(unit.BaseTestCase):
    """Test for V3 Limits API validation."""

    def setUp(self):
        super(LimitValidationTestCase, self).setUp()
        create_registered_limits = limit_schema.registered_limit_create
        update_registered_limits = limit_schema.registered_limit_update
        create_limits = limit_schema.limit_create
        update_limits = limit_schema.limit_update
        self.create_registered_limits_validator = validators.SchemaValidator(create_registered_limits)
        self.update_registered_limits_validator = validators.SchemaValidator(update_registered_limits)
        self.create_limits_validator = validators.SchemaValidator(create_limits)
        self.update_limits_validator = validators.SchemaValidator(update_limits)

    def test_validate_registered_limit_create_request_succeeds(self):
        request_to_validate = [{'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'description': 'test description'}]
        self.create_registered_limits_validator.validate(request_to_validate)

    def test_validate_registered_limit_create_request_without_optional(self):
        request_to_validate = [{'service_id': uuid.uuid4().hex, 'resource_name': 'volume', 'default_limit': 10}]
        self.create_registered_limits_validator.validate(request_to_validate)

    def test_validate_registered_limit_update_request_without_region(self):
        request_to_validate = {'service_id': uuid.uuid4().hex, 'resource_name': 'volume', 'default_limit': 10}
        self.update_registered_limits_validator.validate(request_to_validate)

    def test_validate_registered_limit_request_with_no_parameters(self):
        request_to_validate = []
        self.assertRaises(exception.SchemaValidationError, self.create_registered_limits_validator.validate, request_to_validate)

    def test_validate_registered_limit_create_request_with_invalid_input(self):
        _INVALID_FORMATS = [{'service_id': 'fake_id'}, {'region_id': 123}, {'resource_name': 123}, {'resource_name': ''}, {'resource_name': 'a' * 256}, {'default_limit': 'not_int'}, {'default_limit': -10}, {'default_limit': 10000000000000000}, {'description': 123}, {'description': True}]
        for invalid_desc in _INVALID_FORMATS:
            request_to_validate = [{'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'description': 'test description'}]
            request_to_validate[0].update(invalid_desc)
            self.assertRaises(exception.SchemaValidationError, self.create_registered_limits_validator.validate, request_to_validate)

    def test_validate_registered_limit_update_request_with_invalid_input(self):
        _INVALID_FORMATS = [{'service_id': 'fake_id'}, {'region_id': 123}, {'resource_name': 123}, {'resource_name': ''}, {'resource_name': 'a' * 256}, {'default_limit': 'not_int'}, {'default_limit': -10}, {'default_limit': 10000000000000000}, {'description': 123}]
        for invalid_desc in _INVALID_FORMATS:
            request_to_validate = {'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'description': 'test description'}
            request_to_validate.update(invalid_desc)
            self.assertRaises(exception.SchemaValidationError, self.update_registered_limits_validator.validate, request_to_validate)

    def test_validate_registered_limit_create_request_with_addition(self):
        request_to_validate = [{'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'more_key': 'more_value'}]
        self.assertRaises(exception.SchemaValidationError, self.create_registered_limits_validator.validate, request_to_validate)

    def test_validate_registered_limit_update_request_with_addition(self):
        request_to_validate = {'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'more_key': 'more_value'}
        self.assertRaises(exception.SchemaValidationError, self.update_registered_limits_validator.validate, request_to_validate)

    def test_validate_registered_limit_create_request_without_required(self):
        for key in ['service_id', 'resource_name', 'default_limit']:
            request_to_validate = [{'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10}]
            request_to_validate[0].pop(key)
            self.assertRaises(exception.SchemaValidationError, self.create_registered_limits_validator.validate, request_to_validate)

    def test_validate_project_limit_create_request_succeeds(self):
        request_to_validate = [{'project_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
        self.create_limits_validator.validate(request_to_validate)

    def test_validate_domain_limit_create_request_succeeds(self):
        request_to_validate = [{'domain_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
        self.create_limits_validator.validate(request_to_validate)

    def test_validate_limit_create_request_without_optional(self):
        request_to_validate = [{'project_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'resource_name': 'volume', 'resource_limit': 10}]
        self.create_limits_validator.validate(request_to_validate)

    def test_validate_limit_update_request_succeeds(self):
        request_to_validate = {'resource_limit': 10, 'description': 'test description'}
        self.update_limits_validator.validate(request_to_validate)

    def test_validate_limit_update_request_without_optional(self):
        request_to_validate = {'resource_limit': 10}
        self.update_limits_validator.validate(request_to_validate)

    def test_validate_limit_request_with_no_parameters(self):
        request_to_validate = []
        self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_limit_create_request_with_invalid_input(self):
        _INVALID_FORMATS = [{'project_id': 'fake_id'}, {'service_id': 'fake_id'}, {'region_id': 123}, {'resource_name': 123}, {'resource_name': ''}, {'resource_name': 'a' * 256}, {'resource_limit': -10}, {'resource_limit': 10000000000000000}, {'resource_limit': 'not_int'}, {'description': 123}]
        for invalid_attribute in _INVALID_FORMATS:
            request_to_validate = [{'project_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
            request_to_validate[0].update(invalid_attribute)
            self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_limit_create_request_with_invalid_domain(self):
        request_to_validate = [{'domain_id': 'fake_id', 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
        self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_limit_update_request_with_invalid_input(self):
        _INVALID_FORMATS = [{'resource_name': 123}, {'resource_limit': 'not_int'}, {'resource_name': ''}, {'resource_name': 'a' * 256}, {'resource_limit': -10}, {'resource_limit': 10000000000000000}, {'description': 123}]
        for invalid_desc in _INVALID_FORMATS:
            request_to_validate = [{'project_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
            request_to_validate[0].update(invalid_desc)
            self.assertRaises(exception.SchemaValidationError, self.update_limits_validator.validate, request_to_validate)

    def test_validate_limit_create_request_with_addition_input_fails(self):
        request_to_validate = [{'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'more_key': 'more_value'}]
        self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_limit_update_request_with_addition_input_fails(self):
        request_to_validate = {'id': uuid.uuid4().hex, 'resource_limit': 10, 'more_key': 'more_value'}
        self.assertRaises(exception.SchemaValidationError, self.update_limits_validator.validate, request_to_validate)

    def test_validate_project_limit_create_request_without_required_fails(self):
        for key in ['project_id', 'service_id', 'resource_name', 'resource_limit']:
            request_to_validate = [{'project_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10}]
            request_to_validate[0].pop(key)
            self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_domain_limit_create_request_without_required_fails(self):
        for key in ['domain_id', 'service_id', 'resource_name', 'resource_limit']:
            request_to_validate = [{'domain_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10}]
            request_to_validate[0].pop(key)
            self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)

    def test_validate_limit_create_request_with_both_project_and_domain(self):
        request_to_validate = [{'project_id': uuid.uuid4().hex, 'domain_id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
        self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)