from twisted.python import roots
from twisted.web import pages, resource
class NameVirtualHost(resource.Resource):
    """I am a resource which represents named virtual hosts."""
    default = None

    def __init__(self):
        """Initialize."""
        resource.Resource.__init__(self)
        self.hosts = {}

    def listStaticEntities(self):
        return resource.Resource.listStaticEntities(self) + [('Virtual Hosts', VirtualHostCollection(self))]

    def getStaticEntity(self, name):
        if name == 'Virtual Hosts':
            return VirtualHostCollection(self)
        else:
            return resource.Resource.getStaticEntity(self, name)

    def addHost(self, name, resrc):
        """Add a host to this virtual host.

        This will take a host named `name', and map it to a resource
        `resrc'.  For example, a setup for our virtual hosts would be::

            nvh.addHost('divunal.com', divunalDirectory)
            nvh.addHost('www.divunal.com', divunalDirectory)
            nvh.addHost('twistedmatrix.com', twistedMatrixDirectory)
            nvh.addHost('www.twistedmatrix.com', twistedMatrixDirectory)
        """
        self.hosts[name] = resrc

    def removeHost(self, name):
        """Remove a host."""
        del self.hosts[name]

    def _getResourceForRequest(self, request):
        """(Internal) Get the appropriate resource for the given host."""
        hostHeader = request.getHeader(b'host')
        if hostHeader is None:
            return self.default or pages.notFound()
        else:
            host = hostHeader.lower().split(b':', 1)[0]
        return self.hosts.get(host, self.default) or pages.notFound('Not Found', f'host {host.decode('ascii', 'replace')!r} not in vhost map')

    def render(self, request):
        """Implementation of resource.Resource's render method."""
        resrc = self._getResourceForRequest(request)
        return resrc.render(request)

    def getChild(self, path, request):
        """Implementation of resource.Resource's getChild method."""
        resrc = self._getResourceForRequest(request)
        if resrc.isLeaf:
            request.postpath.insert(0, request.prepath.pop(-1))
            return resrc
        else:
            return resrc.getChildWithDefault(path, request)