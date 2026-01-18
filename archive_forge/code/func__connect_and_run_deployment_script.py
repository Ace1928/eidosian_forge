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
def _connect_and_run_deployment_script(self, task, node, ssh_hostname, ssh_port, ssh_username, ssh_password, ssh_key_file, ssh_key_password, ssh_timeout, timeout, max_tries):
    """
        Establish an SSH connection to the node and run the provided deployment
        task.

        :rtype: :class:`.Node`:
        :return: Node instance on success.
        """
    ssh_client = SSHClient(hostname=ssh_hostname, port=ssh_port, username=ssh_username, password=ssh_key_password or ssh_password, key_files=ssh_key_file, timeout=ssh_timeout)
    ssh_client = self._ssh_client_connect(ssh_client=ssh_client, timeout=timeout)
    node = self._run_deployment_script(task=task, node=node, ssh_client=ssh_client, max_tries=max_tries)
    return node