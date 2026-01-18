from twisted.python import roots
from twisted.web import pages, resource
def _getResourceForRequest(self, request):
    """(Internal) Get the appropriate resource for the given host."""
    hostHeader = request.getHeader(b'host')
    if hostHeader is None:
        return self.default or pages.notFound()
    else:
        host = hostHeader.lower().split(b':', 1)[0]
    return self.hosts.get(host, self.default) or pages.notFound('Not Found', f'host {host.decode('ascii', 'replace')!r} not in vhost map')