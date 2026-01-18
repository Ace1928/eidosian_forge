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
class VCloudNodeDriver(NodeDriver):
    """
    vCloud node driver
    """
    type = Provider.VCLOUD
    name = 'vCloud'
    website = 'http://www.vmware.com/products/vcloud/'
    connectionCls = VCloudConnection
    org = None
    _vdcs = None
    NODE_STATE_MAP = {'0': NodeState.PENDING, '1': NodeState.PENDING, '2': NodeState.PENDING, '3': NodeState.PENDING, '4': NodeState.RUNNING}
    features = {'create_node': ['password']}

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, **kwargs):
        if cls is VCloudNodeDriver:
            if api_version == '0.8':
                cls = VCloudNodeDriver
            elif api_version == '1.5':
                cls = VCloud_1_5_NodeDriver
            elif api_version == '5.1':
                cls = VCloud_5_1_NodeDriver
            elif api_version == '5.5':
                cls = VCloud_5_5_NodeDriver
            else:
                raise NotImplementedError('No VCloudNodeDriver found for API version %s' % api_version)
        return super().__new__(cls)

    @property
    def vdcs(self):
        """
        vCloud virtual data centers (vDCs).

        :return: list of vDC objects
        :rtype: ``list`` of :class:`Vdc`
        """
        if not self._vdcs:
            self.connection.check_org()
            res = self.connection.request(self.org)
            self._vdcs = [self._to_vdc(self.connection.request(get_url_path(i.get('href'))).object) for i in res.object.findall(fixxpath(res.object, 'Link')) if i.get('type') == 'application/vnd.vmware.vcloud.vdc+xml']
        return self._vdcs

    def _to_vdc(self, vdc_elm):
        return Vdc(vdc_elm.get('href'), vdc_elm.get('name'), self)

    def _get_vdc(self, vdc_name):
        vdc = None
        if not vdc_name:
            vdc = self.vdcs[0]
        else:
            for v in self.vdcs:
                if v.name == vdc_name or v.id == vdc_name:
                    vdc = v
            if vdc is None:
                raise ValueError('%s virtual data centre could not be found' % vdc_name)
        return vdc

    @property
    def networks(self):
        networks = []
        for vdc in self.vdcs:
            res = self.connection.request(get_url_path(vdc.id)).object
            networks.extend([network for network in res.findall(fixxpath(res, 'AvailableNetworks/Network'))])
        return networks

    def _to_image(self, image):
        image = NodeImage(id=image.get('href'), name=image.get('name'), driver=self.connection.driver)
        return image

    def _to_node(self, elm):
        state = self.NODE_STATE_MAP[elm.get('status')]
        name = elm.get('name')
        public_ips = []
        private_ips = []
        connections = elm.findall('%s/%s' % ('{http://schemas.dmtf.org/ovf/envelope/1}NetworkConnectionSection', fixxpath(elm, 'NetworkConnection')))
        if not connections:
            connections = elm.findall(fixxpath(elm, 'Children/Vm/NetworkConnectionSection/NetworkConnection'))
        for connection in connections:
            ips = [ip.text for ip in connection.findall(fixxpath(elm, 'IpAddress'))]
            if connection.get('Network') == 'Internal':
                private_ips.extend(ips)
            else:
                public_ips.extend(ips)
        node = Node(id=elm.get('href'), name=name, state=state, public_ips=public_ips, private_ips=private_ips, driver=self.connection.driver)
        return node

    def _get_catalog_hrefs(self):
        res = self.connection.request(self.org)
        catalogs = [i.get('href') for i in res.object.findall(fixxpath(res.object, 'Link')) if i.get('type') == 'application/vnd.vmware.vcloud.catalog+xml']
        return catalogs

    def _wait_for_task_completion(self, task_href, timeout=DEFAULT_TASK_COMPLETION_TIMEOUT):
        start_time = time.time()
        res = self.connection.request(get_url_path(task_href))
        status = res.object.get('status')
        while status != 'success':
            if status == 'error':
                error_elem = res.object.find(fixxpath(res.object, 'Error'))
                error_msg = 'Unknown error'
                if error_elem is not None:
                    error_msg = error_elem.get('message')
                raise Exception('Error status returned by task {}.: {}'.format(task_href, error_msg))
            if status == 'canceled':
                raise Exception('Canceled status returned by task %s.' % task_href)
            if time.time() - start_time >= timeout:
                raise Exception('Timeout ({} sec) while waiting for task {}.'.format(timeout, task_href))
            time.sleep(5)
            res = self.connection.request(get_url_path(task_href))
            status = res.object.get('status')

    def destroy_node(self, node):
        node_path = get_url_path(node.id)
        try:
            res = self.connection.request('%s/power/action/poweroff' % node_path, method='POST')
            self._wait_for_task_completion(res.object.get('href'))
        except Exception:
            pass
        try:
            res = self.connection.request('%s/action/undeploy' % node_path, method='POST')
            self._wait_for_task_completion(res.object.get('href'))
        except ExpatError:
            pass
        except Exception:
            pass
        res = self.connection.request(node_path, method='DELETE')
        return res.status == httplib.ACCEPTED

    def reboot_node(self, node):
        res = self.connection.request('%s/power/action/reset' % get_url_path(node.id), method='POST')
        return res.status in [httplib.ACCEPTED, httplib.NO_CONTENT]

    def list_nodes(self):
        return self.ex_list_nodes()

    def ex_list_nodes(self, vdcs=None):
        """
        List all nodes across all vDCs. Using 'vdcs' you can specify which vDCs
        should be queried.

        :param vdcs: None, vDC or a list of vDCs to query. If None all vDCs
                     will be queried.
        :type vdcs: :class:`Vdc`

        :rtype: ``list`` of :class:`Node`
        """
        if not vdcs:
            vdcs = self.vdcs
        if not isinstance(vdcs, (list, tuple)):
            vdcs = [vdcs]
        nodes = []
        for vdc in vdcs:
            res = self.connection.request(get_url_path(vdc.id))
            elms = res.object.findall(fixxpath(res.object, 'ResourceEntities/ResourceEntity'))
            vapps = [(i.get('name'), i.get('href')) for i in elms if i.get('type') == 'application/vnd.vmware.vcloud.vApp+xml' and i.get('name')]
            for vapp_name, vapp_href in vapps:
                try:
                    res = self.connection.request(get_url_path(vapp_href), headers={'Content-Type': 'application/vnd.vmware.vcloud.vApp+xml'})
                    nodes.append(self._to_node(res.object))
                except Exception as e:
                    if not (e.args[0].tag.endswith('Error') and e.args[0].get('minorErrorCode') == 'ACCESS_TO_RESOURCE_IS_FORBIDDEN'):
                        raise
        return nodes

    def _to_size(self, ram):
        ns = NodeSize(id=None, name='%s Ram' % ram, ram=ram, disk=None, bandwidth=None, price=None, driver=self.connection.driver)
        return ns

    def list_sizes(self, location=None):
        sizes = [self._to_size(i) for i in VIRTUAL_MEMORY_VALS]
        return sizes

    def _get_catalogitems_hrefs(self, catalog):
        """Given a catalog href returns contained catalog item hrefs"""
        res = self.connection.request(get_url_path(catalog), headers={'Content-Type': 'application/vnd.vmware.vcloud.catalog+xml'}).object
        cat_items = res.findall(fixxpath(res, 'CatalogItems/CatalogItem'))
        cat_item_hrefs = [i.get('href') for i in cat_items if i.get('type') == 'application/vnd.vmware.vcloud.catalogItem+xml']
        return cat_item_hrefs

    def _get_catalogitem(self, catalog_item):
        """Given a catalog item href returns elementree"""
        res = self.connection.request(get_url_path(catalog_item), headers={'Content-Type': 'application/vnd.vmware.vcloud.catalogItem+xml'}).object
        return res

    def list_images(self, location=None):
        images = []
        for vdc in self.vdcs:
            res = self.connection.request(get_url_path(vdc.id)).object
            res_ents = res.findall(fixxpath(res, 'ResourceEntities/ResourceEntity'))
            images += [self._to_image(i) for i in res_ents if i.get('type') == 'application/vnd.vmware.vcloud.vAppTemplate+xml']
        for catalog in self._get_catalog_hrefs():
            for cat_item in self._get_catalogitems_hrefs(catalog):
                res = self._get_catalogitem(cat_item)
                res_ents = res.findall(fixxpath(res, 'Entity'))
                images += [self._to_image(i) for i in res_ents if i.get('type') == 'application/vnd.vmware.vcloud.vAppTemplate+xml']

        def idfun(image):
            return image.id
        return self._uniquer(images, idfun)

    def _uniquer(self, seq, idfun=None):
        if idfun is None:

            def idfun(x):
                return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen:
                continue
            seen[marker] = 1
            result.append(item)
        return result

    def create_node(self, name, size, image, auth=None, ex_network=None, ex_vdc=None, ex_cpus=1, ex_row=None, ex_group=None):
        """
        Creates and returns node.

        :keyword    ex_network: link to a "Network" e.g.,
                    ``https://services.vcloudexpress...``
        :type       ex_network: ``str``

        :keyword    ex_vdc: Name of organisation's virtual data
                            center where vApp VMs will be deployed.
        :type       ex_vdc: ``str``

        :keyword    ex_cpus: number of virtual cpus (limit depends on provider)
        :type       ex_cpus: ``int``

        :type       ex_row: ``str``

        :type       ex_group: ``str``
        """
        try:
            network = ex_network or self.networks[0].get('href')
        except IndexError:
            network = ''
        password = None
        auth = self._get_and_check_auth(auth)
        password = auth.password
        instantiate_xml = InstantiateVAppXML(name=name, template=image.id, net_href=network, cpus=str(ex_cpus), memory=str(size.ram), password=password, row=ex_row, group=ex_group)
        vdc = self._get_vdc(ex_vdc)
        content_type = 'application/vnd.vmware.vcloud.instantiateVAppTemplateParams+xml'
        res = self.connection.request('%s/action/instantiateVAppTemplate' % get_url_path(vdc.id), data=instantiate_xml.tostring(), method='POST', headers={'Content-Type': content_type})
        vapp_path = get_url_path(res.object.get('href'))
        res = self.connection.request('%s/action/deploy' % vapp_path, method='POST')
        self._wait_for_task_completion(res.object.get('href'))
        res = self.connection.request('%s/power/action/powerOn' % vapp_path, method='POST')
        res = self.connection.request(vapp_path)
        node = self._to_node(res.object)
        if getattr(auth, 'generated', False):
            node.extra['password'] = auth.password
        return node