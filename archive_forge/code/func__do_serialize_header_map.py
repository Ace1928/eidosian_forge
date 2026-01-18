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
def _do_serialize_header_map(self, header_prefix, headers, user_input):
    for key, val in user_input.items():
        full_key = header_prefix + key
        headers[full_key] = val