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
class NodeAuthPassword:
    """
    A password to be used for authentication to a node.
    """

    def __init__(self, password, generated=False):
        """
        :param password: Password.
        :type password: ``str``

        :type generated: ``True`` if this password was automatically generated,
                         ``False`` otherwise.
        """
        self.password = password
        self.generated = generated

    def __repr__(self):
        return '<NodeAuthPassword>'