from io import BytesIO
from ...bzr.smart import medium
from ...transport import chroot, get_transport
from ...urlutils import local_path_to_url
class SmartWSGIApp:
    """A WSGI application for the bzr smart server."""

    def __init__(self, backing_transport, root_client_path='/'):
        """Constructor.

        :param backing_transport: a transport.  Requests will be processed
            relative to this transport.
        :param root_client_path: the client path that maps to the root of
            backing_transport.  This is used to interpret relpaths received from
            the client.
        """
        self.chroot_server = chroot.ChrootServer(backing_transport)
        self.chroot_server.start_server()
        self.backing_transport = get_transport(self.chroot_server.get_url())
        self.root_client_path = root_client_path

    def __call__(self, environ, start_response):
        """WSGI application callable."""
        if environ['REQUEST_METHOD'] != 'POST':
            start_response('405 Method not allowed', [('Allow', 'POST')])
            return []
        relpath = environ['breezy.relpath']
        if not relpath.startswith('/'):
            relpath = '/' + relpath
        if not relpath.endswith('/'):
            relpath += '/'
        if relpath.startswith(self.root_client_path):
            adjusted_rcp = None
            adjusted_relpath = relpath[len(self.root_client_path):]
        elif self.root_client_path.startswith(relpath):
            adjusted_rcp = '/' + self.root_client_path[len(relpath):]
            adjusted_relpath = '.'
        else:
            adjusted_rcp = self.root_client_path
            adjusted_relpath = relpath
        if adjusted_relpath.startswith('/'):
            adjusted_relpath = adjusted_relpath[1:]
        if adjusted_relpath.startswith('/'):
            raise AssertionError(adjusted_relpath)
        transport = self.backing_transport.clone(adjusted_relpath)
        out_buffer = BytesIO()
        request_data_length = int(environ['CONTENT_LENGTH'])
        request_data_bytes = environ['wsgi.input'].read(request_data_length)
        smart_protocol_request = self.make_request(transport, out_buffer.write, request_data_bytes, adjusted_rcp)
        if smart_protocol_request.next_read_size() != 0:
            response_data = b'error\x01incomplete request\n'
        else:
            response_data = out_buffer.getvalue()
        headers = [('Content-type', 'application/octet-stream')]
        headers.append(('Content-Length', str(len(response_data))))
        start_response('200 OK', headers)
        return [response_data]

    def make_request(self, transport, write_func, request_bytes, rcp):
        protocol_factory, unused_bytes = medium._get_protocol_factory_for_bytes(request_bytes)
        server_protocol = protocol_factory(transport, write_func, rcp, self.backing_transport)
        server_protocol.accept_bytes(unused_bytes)
        return server_protocol