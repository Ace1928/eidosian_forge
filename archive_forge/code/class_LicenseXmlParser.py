from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
class LicenseXmlParser(object):

    def __init__(self, content=None):
        self.raw_content = content
        try:
            self.content = xml.etree.ElementTree.fromstring(content)
        except xml.etree.ElementTree.ParseError as ex:
            raise F5ModuleError("Provided XML payload is invalid. Received '{0}'.".format(str(ex)))

    @property
    def namespaces(self):
        result = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
        return result

    @property
    def eula(self):
        try:
            root = self.content.findall('.//eula', self.namespaces)
            return root[0].text
        except Exception:
            return None

    @property
    def license(self):
        try:
            root = self.content.findall('.//license', self.namespaces)
            return root[0].text
        except Exception:
            return None

    def find_element(self, value):
        root = self.content.findall('.//multiRef', self.namespaces)
        if len(root) == 0:
            return None
        for elem in root:
            for k, v in iteritems(elem.attrib):
                if value in v:
                    return elem

    @property
    def state(self):
        elem = self.find_element('TransactionState')
        if elem is not None:
            return elem.text

    @property
    def fault_number(self):
        fault = self.get_fault()
        return fault.get('faultNumber', None)

    @property
    def fault_text(self):
        fault = self.get_fault()
        return fault.get('faultText', None)

    def get_fault(self):
        result = dict()
        self.set_result_for_license_fault(result)
        self.set_result_for_general_fault(result)
        if 'faultNumber' not in result:
            result['faultNumber'] = None
        return result

    def set_result_for_license_fault(self, result):
        root = self.find_element('LicensingFault')
        if root is None:
            return result
        for elem in root:
            if elem.tag == 'faultNumber':
                result['faultNumber'] = elem.text
            elif elem.tag == 'faultText':
                tmp = elem.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}nil', None)
                if tmp == 'true':
                    result['faultText'] = None
                else:
                    result['faultText'] = elem.text

    def set_result_for_general_fault(self, result):
        namespaces = {'ns2': 'http://schemas.xmlsoap.org/soap/envelope/'}
        root = self.content.findall('.//ns2:Fault', namespaces)
        if len(root) == 0:
            return None
        for elem in root[0]:
            if elem.tag == 'faultstring':
                result['faultText'] = elem.text

    def json(self):
        result = dict(eula=self.eula or None, license=self.license or None, state=self.state or None, fault_number=self.fault_number, fault_text=self.fault_text or None)
        return result