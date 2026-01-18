import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class Instantiate_1_5_VAppXML:

    def __init__(self, name, template, network, vm_network=None, vm_fence=None, description=None):
        self.name = name
        self.template = template
        self.network = network
        self.vm_network = vm_network
        self.vm_fence = vm_fence
        self.description = description
        self._build_xmltree()

    def tostring(self):
        return ET.tostring(self.root)

    def _build_xmltree(self):
        self.root = self._make_instantiation_root()
        if self.network is not None:
            instantiation_params = ET.SubElement(self.root, 'InstantiationParams')
            network_config_section = ET.SubElement(instantiation_params, 'NetworkConfigSection')
            ET.SubElement(network_config_section, 'Info', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1'})
            network_config = ET.SubElement(network_config_section, 'NetworkConfig')
            self._add_network_association(network_config)
        if self.description is not None:
            ET.SubElement(self.root, 'Description').text = self.description
        self._add_vapp_template(self.root)

    def _make_instantiation_root(self):
        return ET.Element('InstantiateVAppTemplateParams', {'name': self.name, 'deploy': 'false', 'powerOn': 'false', 'xml:lang': 'en', 'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})

    def _add_vapp_template(self, parent):
        return ET.SubElement(parent, 'Source', {'href': self.template})

    def _add_network_association(self, parent):
        if self.vm_network is None:
            parent.set('networkName', self.network.get('name'))
        else:
            parent.set('networkName', self.vm_network)
        configuration = ET.SubElement(parent, 'Configuration')
        ET.SubElement(configuration, 'ParentNetwork', {'href': self.network.get('href')})
        if self.vm_fence is None:
            fencemode = self.network.find(fixxpath(self.network, 'Configuration/FenceMode')).text
        else:
            fencemode = self.vm_fence
        ET.SubElement(configuration, 'FenceMode').text = fencemode