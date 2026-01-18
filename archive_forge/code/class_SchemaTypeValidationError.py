from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
class SchemaTypeValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_type):
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected type: {expected_type}\n                Received value of type {type(value)}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_type):
        if not isinstance(value, expected_type):
            raise SchemaTypeValidationError(value, full_schema, path, expected_type)