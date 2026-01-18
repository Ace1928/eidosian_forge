from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import socket
import string
import time
from dns import resolver
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
import six
def _GetSourceIPAddress(self):
    """Get current external IP from Google DNS server.

    Returns:
      str, an ipv4 address represented by string
    """
    re = resolver.Resolver()
    re.nameservers = [socket.gethostbyname('ns1.google.com')]
    for rdata in re.query(qname='o-o.myaddr.l.google.com', rdtype='TXT'):
        return six.text_type(rdata).strip('"')
    return ''