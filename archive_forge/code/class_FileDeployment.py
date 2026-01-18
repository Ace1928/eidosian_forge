import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
class FileDeployment(Deployment):
    """
    Installs a file on the server.
    """

    def __init__(self, source, target):
        """
        :type source: ``str``
        :keyword source: Local path of file to be installed

        :type target: ``str``
        :keyword target: Path to install file on node
        """
        self.source = source
        self.target = target

    def run(self, node, client):
        """
        Upload the file, retaining permissions.

        See also :class:`Deployment.run`
        """
        perms = int(oct(os.stat(self.source).st_mode)[4:], 8)
        with open(self.source, 'rb') as fp:
            client.putfo(path=self.target, chmod=perms, fo=fp)
        return node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<FileDeployment source={}, target={}>'.format(self.source, self.target)