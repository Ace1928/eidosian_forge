from openstack import resource
from openstack import utils
class ServerRemoteConsole(resource.Resource):
    resource_key = 'remote_console'
    base_path = '/servers/%(server_id)s/remote-consoles'
    allow_create = True
    allow_fetch = False
    allow_commit = False
    allow_delete = False
    allow_list = False
    _max_microversion = '2.8'
    protocol = resource.Body('protocol')
    type = resource.Body('type')
    url = resource.Body('url')
    server_id = resource.URI('server_id')

    def create(self, session, prepend_key=True, base_path=None, **params):
        if not self.protocol:
            self.protocol = CONSOLE_TYPE_PROTOCOL_MAPPING.get(self.type)
        if not utils.supports_microversion(session, '2.8') and self.type == 'webmks':
            raise ValueError('Console type webmks is not supported on server side')
        return super(ServerRemoteConsole, self).create(session, prepend_key=prepend_key, base_path=base_path, **params)