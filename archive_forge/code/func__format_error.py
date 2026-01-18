import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _format_error(self, error):
    error_type, name, additional = error
    name = self._get_name(name)
    if error_type == 'missing required field':
        return f'Missing required parameter in {name}: "{additional['required_name']}"'
    elif error_type == 'unknown field':
        unknown_param = additional['unknown_param']
        valid_names = ', '.join(additional['valid_names'])
        return f'Unknown parameter in {name}: "{unknown_param}", must be one of: {valid_names}'
    elif error_type == 'invalid type':
        param = additional['param']
        param_type = type(param)
        valid_types = ', '.join(additional['valid_types'])
        return f'Invalid type for parameter {name}, value: {param}, type: {param_type}, valid types: {valid_types}'
    elif error_type == 'invalid range':
        param = additional['param']
        min_allowed = additional['min_allowed']
        return f'Invalid value for parameter {name}, value: {param}, valid min value: {min_allowed}'
    elif error_type == 'invalid length':
        param = additional['param']
        min_allowed = additional['min_allowed']
        return f'Invalid length for parameter {name}, value: {param}, valid min length: {min_allowed}'
    elif error_type == 'unable to encode to json':
        return 'Invalid parameter {} must be json serializable: {}'.format(name, additional['type_error'])
    elif error_type == 'invalid type for document':
        param = additional['param']
        param_type = type(param)
        valid_types = ', '.join(additional['valid_types'])
        return f'Invalid type for document parameter {name}, value: {param}, type: {param_type}, valid types: {valid_types}'
    elif error_type == 'more than one input':
        members = ', '.join(additional['members'])
        return f'Invalid number of parameters set for tagged union structure {name}. Can only set one of the following keys: {members}.'
    elif error_type == 'empty input':
        members = ', '.join(additional['members'])
        return f'Must set one of the following keys for tagged unionstructure {name}: {members}.'