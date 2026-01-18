from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import http.client
import logging
import select
import socket
import ssl
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as iap_utils
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ShouldStop(self):
    """Signals to parent thread that this connection should be closed."""
    return self._stopping