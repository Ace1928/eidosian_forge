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
def _lookup_hist_down(self):
    if self.histindex > 0:
        self.histindex -= 1
        self.buf = self.history[self.histindex]
        self.curpos = self.promptlen + len(self.buf)
        self._refreshline()
    else:
        self._clearline()