import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
class GridscaleNetwork:
    """
    Network Object

    :param id: uuid
    :type id: ``str``
    :param name: Name of Network
    :type name: ``str``
    :param status: Network status
    :type status: ``str``
    :param relations: object related to network
    :type relations: ``object``
    :param create_time: Time Network was created
    :type create_time: ``str``
    """

    def __init__(self, id, name, status, create_time, relations):
        self.id = id
        self.name = name
        self.status = status
        self.create_time = create_time
        self.relations = relations

    def __repr__(self):
        return 'Network: id={}, name={}, status={}, create_time={}, relations={}'.format(self.id, self.name, self.status, self.create_time, self.relations)