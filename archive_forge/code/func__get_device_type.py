import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
def _get_device_type(self):
    return 'socket'