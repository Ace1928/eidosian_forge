import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
def _recursive_dict(self, element):
    return (element.tag, dict(map(self._recursive_dict, element)) or element.text)