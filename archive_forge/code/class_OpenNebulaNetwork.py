import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebulaNetwork:
    """
    Provide a common interface for handling networks of all types.

    Network objects are analogous to physical switches connecting two or
    more physical nodes together. The Network object provides the interface in
    libcloud through which we can manipulate networks in different cloud
    providers in the same way. Network objects don't actually do much directly
    themselves, instead the network driver handles the connection to the
    network.

    You don't normally create a network object yourself; instead you use
    a driver and then have that create the network for you.

    >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
    >>> driver = DummyNodeDriver()
    >>> network = driver.create_network()
    >>> network = driver.list_networks()[0]
    >>> network.name
    'dummy-1'
    """

    def __init__(self, id, name, address, size, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.address = address
        self.size = size
        self.driver = driver
        self.uuid = self.get_uuid()
        self.extra = extra or {}

    def get_uuid(self):
        """
        Unique hash for this network.

        The hash is a function of an SHA1 hash of the network's ID and
        its driver which means that it should be unique between all
        networks. In some subclasses there is no ID available so the
        public IP address is used. This means that, unlike a properly
        done system UUID, the same UUID may mean a different system
        install at a different time

        >>> from libcloud.network.drivers.dummy import DummyNetworkDriver
        >>> driver = DummyNetworkDriver()
        >>> network = driver.create_network()
        >>> network.get_uuid()
        'd3748461511d8b9b0e0bfa0d4d3383a619a2bb9f'

        Note, for example, that this example will always produce the
        same UUID!

        :rtype:  ``str``
        :return: Unique identifier for this instance.
        """
        return hashlib.sha1(b('{}:{}'.format(self.id, self.driver.type))).hexdigest()

    def __repr__(self):
        return '<OpenNebulaNetwork: uuid=%s, name=%s, address=%s, size=%s, provider=%s ...>' % (self.uuid, self.name, self.address, self.size, self.driver.name)