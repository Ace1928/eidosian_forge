from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def RegisterDebuggee(self, description, uniquifier, agent_version=None):
    """Register a debuggee with the Cloud Debugger.

    This method is primarily intended to simplify testing, since it registering
    a debuggee is only a small part of the functionality of a debug agent, and
    the rest of the API is not supported here.
    Args:
      description: A concise description of the debuggee.
      uniquifier: A string uniquely identifying the debug target. Note that the
        uniquifier distinguishes between different deployments of a service,
        not between different replicas of a single deployment. I.e., all
        replicas of a single deployment should report the same uniquifier.
      agent_version: A string describing the program registering the debuggee.
        Defaults to "google.com/gcloud/NNN" where NNN is the gcloud version.
    Returns:
      The registered Debuggee.
    """
    if not agent_version:
        agent_version = self.CLIENT_VERSION
    request = self._debug_messages.RegisterDebuggeeRequest(debuggee=self._debug_messages.Debuggee(project=self._project, description=description, uniquifier=uniquifier, agentVersion=agent_version))
    try:
        response = self._debug_client.controller_debuggees.Register(request)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)
    return Debuggee(response.debuggee)