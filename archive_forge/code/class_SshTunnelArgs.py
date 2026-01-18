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
class SshTunnelArgs(object):
    """A class to hold some options for IAP Tunnel SSH/SCP.

  Attributes:
    track: str/None, the prefix of the track for the inner gcloud.
    project: str, the project id (string with dashes).
    zone: str, the zone name.
    instance: str, the instance name (or IP or FQDN for on-prem).
    region: str, the region name (on-prem only).
    network: str, the network name (on-prem only).
    pass_through_args: [str], additional args to be passed to the inner gcloud.
  """

    def __init__(self):
        self.track = None
        self.project = ''
        self.zone = ''
        self.instance = ''
        self.region = ''
        self.network = ''
        self.pass_through_args = []

    def _Members(self):
        return (self.track, self.project, self.zone, self.instance, self.region, self.network, self.pass_through_args)

    def __eq__(self, other):
        return self._Members() == other._Members()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'SshTunnelArgs<%r>' % (self._Members(),)