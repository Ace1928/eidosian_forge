import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _do_detach_volume(self, node_id, disk_id):
    action = ET.Element('ACTION')
    perform = ET.SubElement(action, 'PERFORM')
    perform.text = 'DETACHDISK'
    params = ET.SubElement(action, 'PARAMS')
    ET.SubElement(params, 'DISK', {'id': disk_id})
    xml = ET.tostring(action)
    url = '/compute/%s/action' % node_id
    resp = self.connection.request(url, method='POST', data=xml)
    return resp.status == httplib.ACCEPTED