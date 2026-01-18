import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def _check_special_validation_cases(self, shape):
    if is_json_value_header(shape):
        return self._validate_jsonvalue_string
    if shape.type_name == 'structure' and shape.is_document_type:
        return self._validate_document