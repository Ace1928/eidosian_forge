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
def _change_vm_script(self, vapp_or_vm_id, vm_script, vm_script_text=None):
    if vm_script is None and vm_script_text is None:
        return
    if vm_script_text is not None:
        script = vm_script_text
    else:
        try:
            with open(vm_script) as fp:
                script = fp.read()
        except Exception:
            return
    vms = self._get_vm_elements(vapp_or_vm_id)
    for vm in vms:
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
        try:
            res.object.find(fixxpath(res.object, 'CustomizationScript')).text = script
        except Exception:
            for i, e in enumerate(res.object):
                if e.tag == '{http://www.vmware.com/vcloud/v1.5}ComputerName':
                    break
            e = ET.Element('{http://www.vmware.com/vcloud/v1.5}CustomizationScript')
            e.text = script
            res.object.insert(i, e)
        self._remove_admin_password(res.object)
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))