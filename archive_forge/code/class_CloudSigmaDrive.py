import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigmaDrive(NodeImage):
    """
    Represents a CloudSigma drive.
    """

    def __init__(self, id, name, size, media, status, driver, extra=None):
        """
        :param id: Drive ID.
        :type id: ``str``

        :param name: Drive name.
        :type name: ``str``

        :param size: Drive size (in GBs).
        :type size: ``float``

        :param media: Drive media (cdrom / disk).
        :type media: ``str``

        :param status: Drive status (unmounted / mounted).
        :type status: ``str``
        """
        super().__init__(id=id, name=name, driver=driver, extra=extra)
        self.size = size
        self.media = media
        self.status = status

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<CloudSigmaSize id=%s, name=%s size=%s, media=%s, status=%s>' % (self.id, self.name, self.size, self.media, self.status)