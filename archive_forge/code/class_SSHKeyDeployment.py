import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
class SSHKeyDeployment(Deployment):
    """
    Installs a public SSH Key onto a server.
    """

    def __init__(self, key):
        """
        :type key: ``str`` or :class:`File` object
        :keyword key: Contents of the public key write or a file object which
                      can be read.
        """
        self.key = self._get_string_value(argument_name='key', argument_value=key)

    def run(self, node, client):
        """
        Installs SSH key into ``.ssh/authorized_keys``

        See also :class:`Deployment.run`
        """
        client.put('.ssh/authorized_keys', contents=self.key, mode='a')
        return node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        key = self.key[:100]
        return '<SSHKeyDeployment key=%s...>' % key