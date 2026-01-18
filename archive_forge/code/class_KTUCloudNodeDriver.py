from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.providers import Provider
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver
class KTUCloudNodeDriver(CloudStackNodeDriver):
    """Driver for KTUCloud Compute platform."""
    EMPTY_DISKOFFERINGID = '0'
    type = Provider.KTUCLOUD
    name = 'KTUCloud'
    website = 'https://ucloudbiz.olleh.com/'

    def list_images(self, location=None):
        args = {'templatefilter': 'executable'}
        if location is not None:
            args['zoneid'] = location.id
        imgs = self._sync_request(command='listAvailableProductTypes', method='GET')
        images = []
        for img in imgs['producttypes']:
            images.append(NodeImage(img['serviceofferingid'], img['serviceofferingdesc'], self, {'hypervisor': '', 'format': '', 'os': img['templatedesc'], 'templateid': img['templateid'], 'zoneid': img['zoneid']}))
        return images

    def list_sizes(self, location=None):
        szs = self._sync_request('listAvailableProductTypes')
        sizes = []
        for sz in szs['producttypes']:
            diskofferingid = sz.get('diskofferingid', self.EMPTY_DISKOFFERINGID)
            sizes.append(NodeSize(diskofferingid, sz['diskofferingdesc'], 0, 0, 0, 0, self))
        return sizes

    def create_node(self, name, size, image, location=None, ex_usageplantype='hourly'):
        params = {'displayname': name, 'serviceofferingid': image.id, 'templateid': str(image.extra['templateid']), 'zoneid': str(image.extra['zoneid'])}
        if ex_usageplantype is None:
            params['usageplantype'] = 'hourly'
        else:
            params['usageplantype'] = ex_usageplantype
        if size.id != self.EMPTY_DISKOFFERINGID:
            params['diskofferingid'] = size.id
        result = self._async_request(command='deployVirtualMachine', params=params, method='GET')
        node = result['virtualmachine']
        return Node(id=node['id'], name=node['displayname'], state=self.NODE_STATE_MAP[node['state']], public_ips=[], private_ips=[], driver=self, extra={'zoneid': image.extra['zoneid'], 'ip_addresses': [], 'forwarding_rules': []})