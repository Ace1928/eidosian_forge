import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebula_3_6_NodeDriver(OpenNebula_3_2_NodeDriver):
    """
    OpenNebula.org node driver for OpenNebula.org v3.6.
    """
    name = 'OpenNebula (v3.6)'

    def create_volume(self, size, name, location=None, snapshot=None):
        storage = ET.Element('STORAGE')
        vol_name = ET.SubElement(storage, 'NAME')
        vol_name.text = name
        vol_type = ET.SubElement(storage, 'TYPE')
        vol_type.text = 'DATABLOCK'
        description = ET.SubElement(storage, 'DESCRIPTION')
        description.text = 'Attached storage'
        public = ET.SubElement(storage, 'PUBLIC')
        public.text = 'NO'
        persistent = ET.SubElement(storage, 'PERSISTENT')
        persistent.text = 'YES'
        fstype = ET.SubElement(storage, 'FSTYPE')
        fstype.text = 'ext3'
        vol_size = ET.SubElement(storage, 'SIZE')
        vol_size.text = str(size)
        xml = ET.tostring(storage)
        volume = self.connection.request('/storage', {'occixml': xml}, method='POST').object
        return self._to_volume(volume)

    def destroy_volume(self, volume):
        url = '/storage/%s' % str(volume.id)
        resp = self.connection.request(url, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def attach_volume(self, node, volume, device):
        action = ET.Element('ACTION')
        perform = ET.SubElement(action, 'PERFORM')
        perform.text = 'ATTACHDISK'
        params = ET.SubElement(action, 'PARAMS')
        ET.SubElement(params, 'STORAGE', {'href': '/storage/%s' % str(volume.id)})
        target = ET.SubElement(params, 'TARGET')
        target.text = device
        xml = ET.tostring(action)
        url = '/compute/%s/action' % node.id
        resp = self.connection.request(url, method='POST', data=xml)
        return resp.status == httplib.ACCEPTED

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

    def detach_volume(self, volume):
        for node in self.list_nodes():
            if type(node.image) is not list:
                continue
            for disk in node.image:
                if disk.id == volume.id:
                    disk_id = disk.extra['disk_id']
                    return self._do_detach_volume(node.id, disk_id)
        return False

    def list_volumes(self):
        return self._to_volumes(self.connection.request('/storage').object)

    def _to_volume(self, storage):
        return StorageVolume(id=storage.findtext('ID'), name=storage.findtext('NAME'), size=int(storage.findtext('SIZE')), driver=self.connection.driver)

    def _to_volumes(self, object):
        volumes = []
        for storage in object.findall('STORAGE'):
            storage_id = storage.attrib['href'].partition('/storage/')[2]
            volumes.append(self._to_volume(self.connection.request('/storage/%s' % storage_id).object))
        return volumes