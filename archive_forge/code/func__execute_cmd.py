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
def _execute_cmd(self, cmds):
    result, _ = self.root(cmds)
    LOG.debug('result: %s', result)
    if cmds[0] == 'quit':
        self.is_connected = False
        return result.status
    self.prompted = False
    self._startnewline()
    output = result.value.replace('\n', '\n\r').rstrip()
    self.chan.send(output)
    self.prompted = True
    self._startnewline()
    return result.status