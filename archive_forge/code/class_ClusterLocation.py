from typing import List, Optional
from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.container.types import ContainerState
class ClusterLocation:
    """
    A physical location where clusters can be.

    >>> from libcloud.container.drivers.dummy import DummyContainerDriver
    >>> driver = DummyContainerDriver(0)
    >>> location = driver.list_locations()[0]
    >>> location.country
    'US'
    """

    def __init__(self, id, name, country, driver):
        """
        :param id: Location ID.
        :type id: ``str``

        :param name: Location name.
        :type name: ``str``

        :param country: Location country.
        :type country: ``str``

        :param driver: Driver this location belongs to.
        :type driver: :class:`.ContainerDriver`
        """
        self.id = str(id)
        self.name = name
        self.country = country
        self.driver = driver

    def __repr__(self):
        return '<ClusterLocation: id=%s, name=%s, country=%s, driver=%s>' % (self.id, self.name, self.country, self.driver.name)