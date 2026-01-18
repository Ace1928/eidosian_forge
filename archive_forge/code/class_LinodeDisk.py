from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeDisk(BaseObject):

    def __init__(self, id, state, name, filesystem, driver, size, extra=None):
        super().__init__(id, state, driver)
        self.name = name
        self.size = size
        self.filesystem = filesystem
        self.extra = extra or {}

    def __repr__(self):
        return '<LinodeDisk: id=%s, name=%s, state=%s, size=%s, filesystem=%s, driver=%s ...>' % (self.id, self.name, self.state, self.size, self.filesystem, self.driver.name)