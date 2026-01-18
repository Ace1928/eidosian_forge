import http.client as httplib
import io
import logging
import netaddr
from oslo_utils import timeutils
from oslo_utils import uuidutils
import requests
import suds
from suds import cache
from suds import client
from suds import plugin
import suds.sax.element as element
from suds import transport
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def add_attribute_for_value(self, node):
    """Helper to handle AnyType.

        Suds does not handle AnyType properly. But VI SDK requires type
        attribute to be set when AnyType is used.

        :param node: XML value node
        """
    if node.name == 'value' or node.name == 'val':
        node.set('xsi:type', 'xsd:string')
    if node.name == 'removeKey':
        try:
            int(node.text)
            node.set('xsi:type', 'xsd:int')
        except (ValueError, TypeError):
            node.set('xsi:type', 'xsd:string')