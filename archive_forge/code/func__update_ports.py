import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
def _update_ports(self):
    """Remove closed port ports."""
    self.ports = [port for port in self.ports if not port.closed]