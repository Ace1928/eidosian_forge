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
class NodeLocation:
    """
    A physical location where nodes can be.

    >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
    >>> driver = DummyNodeDriver(0)
    >>> location = driver.list_locations()[0]
    >>> location.country
    'US'
    """

    def __init__(self, id, name, country, driver, extra=None):
        """
        :param id: Location ID.
        :type id: ``str``

        :param name: Location name.
        :type name: ``str``

        :param country: Location country.
        :type country: ``str``

        :param driver: Driver this location belongs to.
        :type driver: :class:`.NodeDriver`

        :param extra: Optional provided specific attributes associated with
                      this location.
        :type extra: ``dict``
        """
        self.id = str(id)
        self.name = name
        self.country = country
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<NodeLocation: id=%s, name=%s, country=%s, driver=%s>' % (self.id, self.name, self.country, self.driver.name)