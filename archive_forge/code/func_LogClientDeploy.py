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