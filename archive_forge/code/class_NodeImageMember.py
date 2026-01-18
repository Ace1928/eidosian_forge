import os
import re
import time
import atexit
import random
import socket
import hashlib
import binascii
import datetime
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Type, Tuple, Union, Callable, Optional
import libcloud.compute.ssh
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import b
from libcloud.common.base import BaseDriver, Connection, ConnectionKey
from libcloud.compute.ssh import SSHClient, BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.common.types import LibcloudError
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet, is_valid_ip_address
class NodeImageMember(UuidMixin):
    """
    A member of an image. At some cloud providers there is a mechanism
    to share images. Once an image is shared with another account that
    user will be a 'member' of the image.

    For example, see the image members schema in the OpenStack Image
    Service API v2 documentation. https://developer.openstack.org/
    api-ref/image/v2/index.html#image-members-schema

    NodeImageMember objects are typically returned by the driver for the
    cloud provider in response to the list_image_members method
    """

    def __init__(self, id, image_id, state, driver, created=None, extra=None):
        """
        :param id: Image member ID.
        :type id: ``str``

        :param id: The associated image ID.
        :type id: ``str``

        :param state: State of the NodeImageMember. If not
                      provided, will default to UNKNOWN.
        :type state: :class:`.NodeImageMemberState`

        :param driver: Driver this image belongs to.
        :type driver: :class:`.NodeDriver`

        :param      created: A datetime object that represents when the
                             image member was created
        :type       created: ``datetime.datetime``

        :param extra: Optional provided specific attributes associated with
                      this image.
        :type extra: ``dict``
        """
        self.id = str(id)
        self.image_id = str(image_id)
        self.state = state
        self.driver = driver
        self.created = created
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<NodeImageMember: id=%s, image_id=%s, state=%s, driver=%s  ...>' % (self.id, self.image_id, self.state, self.driver.name)