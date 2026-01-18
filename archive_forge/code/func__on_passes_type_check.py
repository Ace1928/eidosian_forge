import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _on_passes_type_check(self, param, shape, errors, name):
    if _type_check(param, errors, name):
        return func(self, param, shape, errors, name)