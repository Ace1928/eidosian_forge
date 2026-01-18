import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
@type_check(valid_types=(dict,))
def _validate_map(self, param, shape, errors, name):
    key_shape = shape.key
    value_shape = shape.value
    for key, value in param.items():
        self._validate(key, key_shape, errors, f'{name} (key: {key})')
        self._validate(value, value_shape, errors, f'{name}.{key}')