import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
@type_check(valid_types=(dict,))
def _validate_structure(self, params, shape, errors, name):
    if shape.is_tagged_union:
        if len(params) == 0:
            errors.report(name, 'empty input', members=shape.members)
        elif len(params) > 1:
            errors.report(name, 'more than one input', members=shape.members)
    for required_member in shape.metadata.get('required', []):
        if required_member not in params:
            errors.report(name, 'missing required field', required_name=required_member, user_params=params)
    members = shape.members
    known_params = []
    for param in params:
        if param not in members:
            errors.report(name, 'unknown field', unknown_param=param, valid_names=list(members))
        else:
            known_params.append(param)
    for param in known_params:
        self._validate(params[param], shape.members[param], errors, f'{name}.{param}')