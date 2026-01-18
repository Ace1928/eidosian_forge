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
def _change_vm_ipmode(self, vapp_or_vm_id, vm_ipmode):
    if vm_ipmode is None:
        return
    vms = self._get_vm_elements(vapp_or_vm_id)
    for vm in vms:
        res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')))
        net_conns = res.object.findall(fixxpath(res.object, 'NetworkConnection'))
        for c in net_conns:
            c.find(fixxpath(c, 'IpAddressAllocationMode')).text = vm_ipmode
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.networkConnectionSection+xml'}
        res = self.connection.request('%s/networkConnectionSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))