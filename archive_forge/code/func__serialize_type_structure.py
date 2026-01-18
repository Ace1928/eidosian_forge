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
def _serialize_type_structure(self, xmlnode, params, shape, name):
    structure_node = ElementTree.SubElement(xmlnode, name)
    if 'xmlNamespace' in shape.serialization:
        namespace_metadata = shape.serialization['xmlNamespace']
        attribute_name = 'xmlns'
        if namespace_metadata.get('prefix'):
            attribute_name += ':%s' % namespace_metadata['prefix']
        structure_node.attrib[attribute_name] = namespace_metadata['uri']
    for key, value in params.items():
        member_shape = shape.members[key]
        member_name = member_shape.serialization.get('name', key)
        if value is None:
            return
        if member_shape.serialization.get('xmlAttribute'):
            xml_attribute_name = member_shape.serialization['name']
            structure_node.attrib[xml_attribute_name] = value
            continue
        self._serialize(member_shape, value, structure_node, member_name)