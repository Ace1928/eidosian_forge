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
class NodeImage(UuidMixin):
    """
    An operating system image.

    NodeImage objects are typically returned by the driver for the
    cloud provider in response to the list_images function

    >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
    >>> driver = DummyNodeDriver(0)
    >>> image = driver.list_images()[0]
    >>> image.name
    'Ubuntu 9.10'

    Apart from name and id, there is no further standard information;
    other parameters are stored in a driver specific "extra" variable

    When creating a node, a node image should be given as an argument
    to the create_node function to decide which OS image to use.

    >>> node = driver.create_node(image=image)
    """

    def __init__(self, id, name, driver, extra=None):
        """
        :param id: Image ID.
        :type id: ``str``

        :param name: Image name.
        :type name: ``str``

        :param driver: Driver this image belongs to.
        :type driver: :class:`.NodeDriver`

        :param extra: Optional provided specific attributes associated with
                      this image.
        :type extra: ``dict``
        """
        self.id = str(id)
        self.name = name
        self.driver = driver
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<NodeImage: id=%s, name=%s, driver=%s  ...>' % (self.id, self.name, self.driver.name)