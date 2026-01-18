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
def _serialize_type_map(self, xmlnode, params, shape, name):
    node = ElementTree.SubElement(xmlnode, name)
    for key, value in params.items():
        entry_node = ElementTree.SubElement(node, 'entry')
        key_name = self._get_serialized_name(shape.key, default_name='key')
        val_name = self._get_serialized_name(shape.value, default_name='value')
        self._serialize(shape.key, key, entry_node, key_name)
        self._serialize(shape.value, value, entry_node, val_name)