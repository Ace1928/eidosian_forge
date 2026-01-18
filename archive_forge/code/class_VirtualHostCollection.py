from twisted.python import roots
from twisted.web import pages, resource
class VirtualHostCollection(roots.Homogenous):
    """Wrapper for virtual hosts collection.

    This exists for configuration purposes.
    """
    entityType = resource.Resource

    def __init__(self, nvh):
        self.nvh = nvh

    def listStaticEntities(self):
        return self.nvh.hosts.items()

    def getStaticEntity(self, name):
        return self.nvh.hosts.get(self)

    def reallyPutEntity(self, name, entity):
        self.nvh.addHost(name, entity)

    def delEntity(self, name):
        self.nvh.removeHost(name)