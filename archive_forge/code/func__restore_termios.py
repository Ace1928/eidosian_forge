import ast
import cmd
import signal
import socket
import sys
import termios
from os_ken import cfg
from os_ken.lib import rpc
@staticmethod
def _restore_termios(t):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, t)