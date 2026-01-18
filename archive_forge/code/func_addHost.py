from twisted.python import roots
from twisted.web import pages, resource
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