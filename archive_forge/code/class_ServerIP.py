from openstack import resource
from openstack import utils
class ServerIP(resource.Resource):
    resources_key = 'addresses'
    base_path = '/servers/%(server_id)s/ips'
    allow_list = True
    address = resource.Body('addr')
    network_label = resource.URI('network_label')
    server_id = resource.URI('server_id')
    version = resource.Body('version')

    @classmethod
    def list(cls, session, paginated=False, server_id=None, network_label=None, base_path=None, **params):
        if base_path is None:
            base_path = cls.base_path
        url = base_path % {'server_id': server_id}
        if network_label is not None:
            url = utils.urljoin(url, network_label)
        resp = session.get(url)
        resp = resp.json()
        if network_label is None:
            resp = resp[cls.resources_key]
        for label, addresses in resp.items():
            for address in addresses:
                yield cls.existing(network_label=label, address=address['addr'], version=address['version'])