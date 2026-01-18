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
def _refreshnewline(self, prompt=None):
    if not prompt and self.prompted:
        prompt = self.PROMPT
    buf = copy(self.buf)
    curpos = copy(self.curpos)
    self._startnewline(prompt)
    self.chan.send(''.join(buf))
    self.buf = buf
    self.curpos = curpos
    self._movcursor(curpos)