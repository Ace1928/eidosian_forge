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
def _get_base64(self, value):
    if isinstance(value, str):
        value = value.encode(self.DEFAULT_ENCODING)
    return base64.b64encode(value).strip().decode(self.DEFAULT_ENCODING)