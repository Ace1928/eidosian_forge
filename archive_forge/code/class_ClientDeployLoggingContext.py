from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
class ClientDeployLoggingContext(object):
    """Context for sending and recording server rpc requests.

  Attributes:
    rpcserver: The AbstractRpcServer to use for the upload.
    requests: A list of client_deployinfo.Request objects to include
      with the client deploy log.
    time_func: Function to get the current time in milliseconds.
    request_params: A dictionary with params to append to requests
  """

    def __init__(self, rpcserver, request_params, usage_reporting, time_func=time.time):
        """Creates a new AppVersionUpload.

    Args:
      rpcserver: The RPC server to use. Should be an instance of HttpRpcServer
        or TestRpcServer.
      request_params: A dictionary with params to append to requests
      usage_reporting: Whether to actually upload data.
      time_func: Function to return the current time in millisecods
        (default time.time).
    """
        self.rpcserver = rpcserver
        self.request_params = request_params
        self.usage_reporting = usage_reporting
        self.time_func = time_func
        self.requests = []

    def Send(self, url, payload='', **kwargs):
        """Sends a request to the server, with common params."""
        start_time_usec = self.GetCurrentTimeUsec()
        request_size_bytes = len(payload)
        try:
            log.debug('Send: {0}, params={1}'.format(url, self.request_params))
            kwargs.update(self.request_params)
            result = self.rpcserver.Send(url, payload=payload, **kwargs)
            self._RegisterReqestForLogging(url, 200, start_time_usec, request_size_bytes)
            return result
        except RPCError as err:
            self._RegisterReqestForLogging(url, err.url_error.code, start_time_usec, request_size_bytes)
            raise

    def GetCurrentTimeUsec(self):
        """Returns the current time in microseconds."""
        return int(round(self.time_func() * 1000 * 1000))

    def _RegisterReqestForLogging(self, path, response_code, start_time_usec, request_size_bytes):
        """Registers a request for client deploy logging purposes."""
        end_time_usec = self.GetCurrentTimeUsec()
        self.requests.append(client_deployinfo.Request(path=path, response_code=response_code, start_time_usec=start_time_usec, end_time_usec=end_time_usec, request_size_bytes=request_size_bytes))

    def LogClientDeploy(self, runtime, start_time_usec, success):
        """Logs a client deployment attempt.

    Args:
      runtime: The runtime for the app being deployed.
      start_time_usec: The start time of the deployment in micro seconds.
      success: True if the deployment succeeded otherwise False.
    """
        if not self.usage_reporting:
            log.info('Skipping usage reporting.')
            return
        end_time_usec = self.GetCurrentTimeUsec()
        try:
            info = client_deployinfo.ClientDeployInfoExternal(runtime=runtime, start_time_usec=start_time_usec, end_time_usec=end_time_usec, requests=self.requests, success=success, sdk_version=config.CLOUD_SDK_VERSION)
            self.Send('/api/logclientdeploy', info.ToYAML())
        except BaseException as e:
            log.debug('Exception logging deploy info continuing - {0}'.format(e))