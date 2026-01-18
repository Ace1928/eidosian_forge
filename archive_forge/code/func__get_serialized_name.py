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
def _get_serialized_name(self, shape, default_name):
    if 'queryName' in shape.serialization:
        return shape.serialization['queryName']
    elif 'name' in shape.serialization:
        name = shape.serialization['name']
        return name[0].upper() + name[1:]
    else:
        return default_name