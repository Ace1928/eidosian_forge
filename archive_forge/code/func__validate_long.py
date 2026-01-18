import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
@type_check(valid_types=(int,))
def _validate_long(self, param, shape, errors, name):
    range_check(name, param, shape, 'invalid range', errors)