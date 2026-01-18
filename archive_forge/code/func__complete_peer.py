import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def _complete_peer(self, text, line, _begidx, _endidx):
    if len((line + 'x').split()) >= 3:
        return []
    return [name for name in peers if name.startswith(text)]