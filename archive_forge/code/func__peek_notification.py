import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
def _peek_notification(self):
    for k, p in peers.items():
        if p.client:
            try:
                p.client.peek_notification()
            except EOFError:
                p.client = None
                print('disconnected %s' % k)