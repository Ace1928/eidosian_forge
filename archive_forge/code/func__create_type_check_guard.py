import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _create_type_check_guard(func):

    def _on_passes_type_check(self, param, shape, errors, name):
        if _type_check(param, errors, name):
            return func(self, param, shape, errors, name)

    def _type_check(param, errors, name):
        if not isinstance(param, valid_types):
            valid_type_names = [str(t) for t in valid_types]
            errors.report(name, 'invalid type', param=param, valid_types=valid_type_names)
            return False
        return True
    return _on_passes_type_check