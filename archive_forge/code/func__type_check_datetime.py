import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _type_check_datetime(self, value):
    try:
        parse_to_aware_datetime(value)
        return True
    except (TypeError, ValueError, AttributeError):
        return False