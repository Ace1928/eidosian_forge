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
def _FetchAccessToken(self, workstation, threaded=False):
    try:
        self.access_token = self.client.projects_locations_workstationClusters_workstationConfigs_workstations.GenerateAccessToken(self.messages.WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGenerateAccessTokenRequest(workstation=workstation)).accessToken
    except Error as e:
        if threaded:
            self.threading_event.set()
        log.error('Error fetching access token: {0}'.format(e))
        sys.exit(1)