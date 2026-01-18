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
def _serialize_type_list(self, xmlnode, params, shape, name):
    member_shape = shape.member
    if shape.serialization.get('flattened'):
        element_name = name
        list_node = xmlnode
    else:
        element_name = member_shape.serialization.get('name', 'member')
        list_node = ElementTree.SubElement(xmlnode, name)
    for item in params:
        self._serialize(member_shape, item, list_node, element_name)