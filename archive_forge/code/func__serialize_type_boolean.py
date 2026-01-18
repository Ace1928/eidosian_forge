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
def _serialize_type_boolean(self, xmlnode, params, shape, name):
    node = ElementTree.SubElement(xmlnode, name)
    if params:
        str_value = 'true'
    else:
        str_value = 'false'
    node.text = str_value