import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def _convert_header_value(self, shape, value):
    if shape.type_name == 'timestamp':
        datetime_obj = parse_to_aware_datetime(value)
        timestamp = calendar.timegm(datetime_obj.utctimetuple())
        timestamp_format = shape.serialization.get('timestampFormat', self.HEADER_TIMESTAMP_FORMAT)
        return self._convert_timestamp_to_str(timestamp, timestamp_format)
    elif shape.type_name == 'list':
        converted_value = [self._convert_header_value(shape.member, v) for v in value if v is not None]
        return ','.join(converted_value)
    elif is_json_value_header(shape):
        return self._get_base64(json.dumps(value, separators=(',', ':')))
    else:
        return value