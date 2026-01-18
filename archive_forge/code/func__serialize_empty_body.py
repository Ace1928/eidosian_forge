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
def _serialize_empty_body(self):
    return b'{}'