from copy import copy
import logging
import os.path
import sys
import paramiko
from os_ken import version
from os_ken.lib import hub
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.root import RootCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
def _handle_esc_seq(self):
    c = self.chan.recv(1)
    c = c.decode()
    if c == '[':
        self._handle_csi_seq()
    else:
        LOG.error('non CSI sequence. do nothing')