from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import socket
import ssl
import sys
import threading
import time
from apitools.base.py.exceptions import Error
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workstations.util import GetClientInstance
from googlecloudsdk.api_lib.workstations.util import GetMessagesModule
from googlecloudsdk.api_lib.workstations.util import VERSION_MAP
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from requests import certs
import six
import websocket
import websocket._exceptions as websocket_exceptions
def _ForwardClientToServer(self, client, server):

    def Forward():
        while True:
            data = client.recv(4096)
            if not data:
                break
            server.send(data)
    t = threading.Thread(target=Forward)
    t.daemon = True
    t.start()