from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
class SchemaKeysValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_keys):
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected keys: {expected_keys}\n                Received value with keys {set(value.keys())}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_keys):
        if set(value.keys()) != set(expected_keys):
            raise SchemaKeysValidationError(value, full_schema, path, expected_keys)