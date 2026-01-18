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
def StartTcpTunnel(self, args, threaded=False):
    """Start a TCP tunnel to a workstation."""
    config_name = args.CONCEPTS.workstation.Parse().Parent().RelativeName()
    try:
        config = self.client.projects_locations_workstationClusters_workstationConfigs.Get(self.messages.WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsGetRequest(name=config_name))
        if hasattr(config, 'disableTcpConnections') and config.disableTcpConnections:
            log.error('TCP tunneling is disabled for workstations under this configuration.')
            sys.exit(1)
    except HttpError:
        pass
    workstation_name = args.CONCEPTS.workstation.Parse().RelativeName()
    try:
        workstation = self.client.projects_locations_workstationClusters_workstationConfigs_workstations.Get(self.messages.WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGetRequest(name=workstation_name))
    except HttpError as e:
        if threaded:
            self.threading_event.set()
        log.error(e)
        sys.exit(1)
    self.host = workstation.host
    self.port = args.workstation_port
    if workstation.state != self.messages.Workstation.StateValueValuesEnum.STATE_RUNNING:
        if threaded:
            self.threading_event.set()
        log.error('Workstation is not running.')
        sys.exit(1)
    self._FetchAccessToken(workstation_name, threaded)
    self._RefreshAccessToken(workstation_name, threaded)
    host, port = self._GetLocalHostPort(args)
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.socket.bind((host, port))
    self.socket.listen(1)
    if port == 0:
        log.status.Print('Picking local unused port [{0}].'.format(self.socket.getsockname()[1]))
    log.status.Print('Listening on port [{0}].'.format(self.socket.getsockname()[1]))
    if threaded:
        self.tcp_tunnel_open = True
        self.threading_event.set()
        while self.tcp_tunnel_open:
            conn, addr = self.socket.accept()
            self._AcceptConnection(conn, addr)
    else:
        try:
            with execution_utils.RaisesKeyboardInterrupt():
                while True:
                    conn, addr = self.socket.accept()
                    self._AcceptConnection(conn, addr)
        except KeyboardInterrupt:
            log.info('Keyboard interrupt received.')
    log.status.Print('Server shutdown complete.')