import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _extract_images(self, compute):
    """
        Extract image disks from a compute node XML representation.

        Extract image disk descriptions from a compute node XML representation,
        converting the disks to an NodeImage object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``list`` of :class:`NodeImage`
        :return: Disks attached to a compute node.
        """
    disks = list()
    for element in compute.findall('DISK'):
        disk = element.find('STORAGE')
        image_id = disk.attrib['href'].partition('/storage/')[2]
        if 'id' in element.attrib:
            disk_id = element.attrib['id']
        else:
            disk_id = None
        disks.append(NodeImage(id=image_id, name=disk.attrib.get('name', None), driver=self.connection.driver, extra={'type': element.findtext('TYPE'), 'disk_id': disk_id, 'target': element.findtext('TARGET')}))
    if len(disks) > 1:
        return disks
    elif len(disks) == 1:
        return disks[0]
    else:
        return None