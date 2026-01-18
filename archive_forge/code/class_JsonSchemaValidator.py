from __future__ import annotations
import os
import fastjsonschema
import jsonschema
from fastjsonschema import JsonSchemaException as _JsonSchemaException
from jsonschema import Draft4Validator as _JsonSchemaValidator
from jsonschema.exceptions import ErrorTree, ValidationError
class JsonSchemaValidator:
    """A json schema validator."""
    name = 'jsonschema'

    def __init__(self, schema):
        """Initialize the validator."""
        self._schema = schema
        self._default_validator = _JsonSchemaValidator(schema)
        self._validator = self._default_validator

    def validate(self, data):
        """Validate incoming data."""
        self._default_validator.validate(data)

    def iter_errors(self, data, schema=None):
        """Iterate over errors in incoming data."""
        if schema is None:
            return self._default_validator.iter_errors(data)
        if hasattr(self._default_validator, 'evolve'):
            return self._default_validator.evolve(schema=schema).iter_errors(data)
        return self._default_validator.iter_errors(data, schema)

    def error_tree(self, errors):
        """Create an error tree for the errors."""
        return ErrorTree(errors=errors)