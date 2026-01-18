import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def _fill_data_to_return_object(self, node, return_obj):
    members = dict(vars(return_obj))
    for name, value in members.items():
        if isinstance(value, _ListOf):
            setattr(return_obj, name, self._fill_list_of(node, value.list_type, value.xml_element_name))
        elif isinstance(value, ScalarListOf):
            setattr(return_obj, name, self._fill_scalar_list_of(node, value.list_type, self._get_serialization_name(name), value.xml_element_name))
        elif isinstance(value, _DictOf):
            setattr(return_obj, name, self._fill_dict_of(node, self._get_serialization_name(name), value.pair_xml_element_name, value.key_xml_element_name, value.value_xml_element_name))
        elif isinstance(value, WindowsAzureData):
            setattr(return_obj, name, self._fill_instance_child(node, name, value.__class__))
        elif isinstance(value, dict):
            setattr(return_obj, name, self._fill_dict(node, self._get_serialization_name(name)))
        elif isinstance(value, _Base64String):
            value = self._fill_data_minidom(node, name, '')
            if value is not None:
                value = self._decode_base64_to_text(value)
            setattr(return_obj, name, value)
        else:
            value = self._fill_data_minidom(node, name, value)
            if value is not None:
                setattr(return_obj, name, value)