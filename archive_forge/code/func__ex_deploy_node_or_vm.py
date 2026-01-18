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
def _ex_deploy_node_or_vm(self, vapp_or_vm_path, ex_force_customization=False):
    data = {'powerOn': 'true', 'forceCustomization': str(ex_force_customization).lower(), 'xmlns': 'http://www.vmware.com/vcloud/v1.5'}
    deploy_xml = ET.Element('DeployVAppParams', data)
    path = get_url_path(vapp_or_vm_path)
    headers = {'Content-Type': 'application/vnd.vmware.vcloud.deployVAppParams+xml'}
    res = self.connection.request('%s/action/deploy' % path, data=ET.tostring(deploy_xml), method='POST', headers=headers)
    self._wait_for_task_completion(res.object.get('href'))