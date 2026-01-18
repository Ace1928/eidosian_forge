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
class TrustValidationTestCase(unit.BaseTestCase):
    """Test for V3 Trust API validation."""
    _valid_roles = [{'name': 'member'}, {'id': uuid.uuid4().hex}, {'id': str(uuid.uuid4())}, {'name': '_member_'}]
    _invalid_roles = [False, True, 123, None]

    def setUp(self):
        super(TrustValidationTestCase, self).setUp()
        create = trust_schema.trust_create
        self.create_trust_validator = validators.SchemaValidator(create)

    def test_validate_trust_succeeds(self):
        """Test that we can validate a trust request."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_all_parameters_succeeds(self):
        """Test that we can validate a trust request with all parameters."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'project_id': uuid.uuid4().hex, 'roles': [{'id': uuid.uuid4().hex}, {'id': uuid.uuid4().hex}], 'expires_at': 'some timestamp', 'remaining_uses': 2}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_without_trustor_id_fails(self):
        """Validate trust request fails without `trustor_id`."""
        request_to_validate = {'trustee_user_id': uuid.uuid4().hex, 'impersonation': False}
        self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_without_trustee_id_fails(self):
        """Validate trust request fails without `trustee_id`."""
        request_to_validate = {'trusor_user_id': uuid.uuid4().hex, 'impersonation': False}
        self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_without_impersonation_fails(self):
        """Validate trust request fails without `impersonation`."""
        request_to_validate = {'trustee_user_id': uuid.uuid4().hex, 'trustor_user_id': uuid.uuid4().hex}
        self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_with_extra_parameters_succeeds(self):
        """Test that we can validate a trust request with extra parameters."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'project_id': uuid.uuid4().hex, 'roles': [{'id': uuid.uuid4().hex}, {'id': uuid.uuid4().hex}], 'expires_at': 'some timestamp', 'remaining_uses': 2, 'extra': 'something extra!'}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_invalid_impersonation_fails(self):
        """Validate trust request with invalid `impersonation` fails."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': 2}
        self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_with_null_remaining_uses_succeeds(self):
        """Validate trust request with null `remaining_uses`."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'remaining_uses': None}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_remaining_uses_succeeds(self):
        """Validate trust request with `remaining_uses` succeeds."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'remaining_uses': 2}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_period_in_user_id_string(self):
        """Validate trust request with a period in the user id string."""
        request_to_validate = {'trustor_user_id': 'john.smith', 'trustee_user_id': 'joe.developer', 'impersonation': False}
        self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_invalid_expires_at_fails(self):
        """Validate trust request with invalid `expires_at` fails."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'expires_at': 3}
        self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_with_role_types_succeeds(self):
        """Validate trust request with `roles` succeeds."""
        for role in self._valid_roles:
            request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'roles': [role]}
            self.create_trust_validator.validate(request_to_validate)

    def test_validate_trust_with_invalid_role_type_fails(self):
        """Validate trust request with invalid `roles` fails."""
        for role in self._invalid_roles:
            request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'roles': role}
            self.assertRaises(exception.SchemaValidationError, self.create_trust_validator.validate, request_to_validate)

    def test_validate_trust_with_list_of_valid_roles_succeeds(self):
        """Validate trust request with a list of valid `roles`."""
        request_to_validate = {'trustor_user_id': uuid.uuid4().hex, 'trustee_user_id': uuid.uuid4().hex, 'impersonation': False, 'roles': self._valid_roles}
        self.create_trust_validator.validate(request_to_validate)