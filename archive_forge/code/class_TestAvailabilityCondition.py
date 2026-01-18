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
class TestAvailabilityCondition(object):

    def test_constructor(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        assert availability_condition.expression == EXPRESSION
        assert availability_condition.title == TITLE
        assert availability_condition.description == DESCRIPTION

    def test_constructor_required_params_only(self):
        availability_condition = make_availability_condition(EXPRESSION)
        assert availability_condition.expression == EXPRESSION
        assert availability_condition.title is None
        assert availability_condition.description is None

    def test_setters(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        availability_condition.expression = OTHER_EXPRESSION
        availability_condition.title = OTHER_TITLE
        availability_condition.description = OTHER_DESCRIPTION
        assert availability_condition.expression == OTHER_EXPRESSION
        assert availability_condition.title == OTHER_TITLE
        assert availability_condition.description == OTHER_DESCRIPTION

    def test_invalid_expression_type(self):
        with pytest.raises(TypeError) as excinfo:
            make_availability_condition([EXPRESSION], TITLE, DESCRIPTION)
        assert excinfo.match('The provided expression is not a string.')

    def test_invalid_title_type(self):
        with pytest.raises(TypeError) as excinfo:
            make_availability_condition(EXPRESSION, False, DESCRIPTION)
        assert excinfo.match('The provided title is not a string or None.')

    def test_invalid_description_type(self):
        with pytest.raises(TypeError) as excinfo:
            make_availability_condition(EXPRESSION, TITLE, False)
        assert excinfo.match('The provided description is not a string or None.')

    def test_to_json_required_params_only(self):
        availability_condition = make_availability_condition(EXPRESSION)
        assert availability_condition.to_json() == {'expression': EXPRESSION}

    def test_to_json_(self):
        availability_condition = make_availability_condition(EXPRESSION, TITLE, DESCRIPTION)
        assert availability_condition.to_json() == {'expression': EXPRESSION, 'title': TITLE, 'description': DESCRIPTION}