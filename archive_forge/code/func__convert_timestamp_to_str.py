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
def _convert_timestamp_to_str(self, value, timestamp_format=None):
    if timestamp_format is None:
        timestamp_format = self.TIMESTAMP_FORMAT
    timestamp_format = timestamp_format.lower()
    datetime_obj = parse_to_aware_datetime(value)
    converter = getattr(self, f'_timestamp_{timestamp_format}')
    final_value = converter(datetime_obj)
    return final_value