from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ctypes
import errno
import functools
import gc
import io
import os
import select
import socket
import sys
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.api_lib.compute import sg_tunnel
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
from six.moves import queue
def AddHostBasedTunnelArgs(parser, support_security_gateway=False):
    """Add the arguments for supporting host-based connections."""
    group = parser.add_argument_group()
    group.add_argument('--region', default=None, required=True, help='Configures the region to use when connecting via IP address or FQDN.')
    if support_security_gateway:
        group_mutex = group.add_argument_group(mutex=True)
        AddSecurityGatewayTunnelArgs(group_mutex.add_argument_group(hidden=True))
        group = group_mutex.add_argument_group()
    AddOnPremTunnelArgs(group)