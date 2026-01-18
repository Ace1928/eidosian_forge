import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
class TestAccessBoundaryRule(object):

    def test_constructor(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        assert access_boundary_rule.available_resource == AVAILABLE_RESOURCE
        assert access_boundary_rule.available_permissions == tuple(AVAILABLE_PERMISSIONS)
        assert access_boundary_rule.availability_condition == availability_condition

    def test_constructor_required_params_only(self):
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS)
        assert access_boundary_rule.available_resource == AVAILABLE_RESOURCE
        assert access_boundary_rule.available_permissions == tuple(AVAILABLE_PERMISSIONS)
        assert access_boundary_rule.availability_condition is None

    def test_setters(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        other_availability_condition = make_availability_condition(OTHER_EXPRESSION, OTHER_TITLE, OTHER_DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        access_boundary_rule.available_resource = OTHER_AVAILABLE_RESOURCE
        access_boundary_rule.available_permissions = OTHER_AVAILABLE_PERMISSIONS
        access_boundary_rule.availability_condition = other_availability_condition
        assert access_boundary_rule.available_resource == OTHER_AVAILABLE_RESOURCE
        assert access_boundary_rule.available_permissions == tuple(OTHER_AVAILABLE_PERMISSIONS)
        assert access_boundary_rule.availability_condition == other_availability_condition

    def test_invalid_available_resource_type(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        with pytest.raises(TypeError) as excinfo:
            make_access_boundary_rule(None, AVAILABLE_PERMISSIONS, availability_condition)
        assert excinfo.match('The provided available_resource is not a string.')

    def test_invalid_available_permissions_type(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        with pytest.raises(TypeError) as excinfo:
            make_access_boundary_rule(AVAILABLE_RESOURCE, [0, 1, 2], availability_condition)
        assert excinfo.match('Provided available_permissions are not a list of strings.')

    def test_invalid_available_permissions_value(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        with pytest.raises(ValueError) as excinfo:
            make_access_boundary_rule(AVAILABLE_RESOURCE, ['roles/storage.objectViewer'], availability_condition)
        assert excinfo.match("available_permissions must be prefixed with 'inRole:'.")

    def test_invalid_availability_condition_type(self):
        with pytest.raises(TypeError) as excinfo:
            make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, {'foo': 'bar'})
        assert excinfo.match("The provided availability_condition is not a 'google.auth.downscoped.AvailabilityCondition' or None.")

    def test_to_json(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS, availability_condition)
        assert access_boundary_rule.to_json() == {'availablePermissions': AVAILABLE_PERMISSIONS, 'availableResource': AVAILABLE_RESOURCE, 'availabilityCondition': {'expression': EXPRESSION, 'title': TITLE, 'description': DESCRIPTION}}

    def test_to_json_required_params_only(self):
        access_boundary_rule = make_access_boundary_rule(AVAILABLE_RESOURCE, AVAILABLE_PERMISSIONS)
        assert access_boundary_rule.to_json() == {'availablePermissions': AVAILABLE_PERMISSIONS, 'availableResource': AVAILABLE_RESOURCE}