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
class CloudSigmaTag:
    """
    Represents a CloudSigma tag object.
    """

    def __init__(self, id, name, resources=None):
        """
        :param id: Tag ID.
        :type id: ``str``

        :param name: Tag name.
        :type name: ``str``

        :param resource: IDs of resources which are associated with this tag.
        :type resources: ``list`` of ``str``
        """
        self.id = id
        self.name = name
        self.resources = resources if resources else []

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<CloudSigmaTag id={}, name={}, resources={}>'.format(self.id, self.name, repr(self.resources))