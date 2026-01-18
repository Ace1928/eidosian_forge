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
class Debugger(DebugObject):
    """Abstracts Cloud Debugger service for a project."""

    def __init__(self, project, debug_client=None, debug_messages=None, resource_client=None, resource_messages=None):
        super(Debugger, self).__init__(debug_client=debug_client, debug_messages=debug_messages, resource_client=resource_client, resource_messages=resource_messages)
        self._project = project

    def ListDebuggees(self, include_inactive=False, include_stale=False):
        """Lists all debug targets registered with the debug service.

    Args:
      include_inactive: If true, also include debuggees that are not currently
        running.
      include_stale: If false, filter out any debuggees that refer to
        stale minor versions. A debugge represents a stale minor version if it
        meets the following criteria:
            1. It has a minorversion label.
            2. All other debuggees with the same name (i.e., all debuggees with
               the same module and version, in the case of app engine) have a
               minorversion label.
            3. The minorversion value for the debuggee is less than the
               minorversion value for at least one other debuggee with the same
               name.
    Returns:
      [Debuggee] A list of debuggees.
    """
        request = self._debug_messages.ClouddebuggerDebuggerDebuggeesListRequest(project=self._project, includeInactive=include_inactive, clientVersion=self.CLIENT_VERSION)
        try:
            response = self._debug_client.debugger_debuggees.List(request)
        except apitools_exceptions.HttpError as error:
            raise errors.UnknownHttpError(error)
        result = [Debuggee(debuggee) for debuggee in response.debuggees]
        if not include_stale:
            return _FilterStaleMinorVersions(result)
        return result

    def DefaultDebuggee(self):
        """Find the default debuggee.

    Returns:
      The default debug target, which is either the only target available
      or the latest minor version of the application, if all targets have the
      same module and version.
    Raises:
      errors.NoDebuggeeError if no debuggee was found.
      errors.MultipleDebuggeesError if there is not a unique default.
    """
        debuggees = self.ListDebuggees()
        if len(debuggees) == 1:
            return debuggees[0]
        if not debuggees:
            raise errors.NoDebuggeeError()
        raise errors.MultipleDebuggeesError(None, debuggees)

    def FindDebuggee(self, pattern=None):
        """Find the unique debuggee matching the given pattern.

    Args:
      pattern: A string containing a debuggee ID or a regular expression that
        matches a single debuggee's name or description. If it matches any
        debuggee name, the description will not be inspected.
    Returns:
      The matching Debuggee.
    Raises:
      errors.MultipleDebuggeesError if the pattern matches multiple debuggees.
      errors.NoDebuggeeError if the pattern matches no debuggees.
    """
        if not pattern:
            debuggee = self.DefaultDebuggee()
            log.status.write('Debug target not specified. Using default target: {0}\n'.format(debuggee.name))
            return debuggee
        try:
            all_debuggees = self.ListDebuggees()
            return self._FilterDebuggeeList(all_debuggees, pattern)
        except errors.NoDebuggeeError:
            pass
        all_debuggees = self.ListDebuggees(include_inactive=True, include_stale=True)
        return self._FilterDebuggeeList(all_debuggees, pattern)

    def _FilterDebuggeeList(self, all_debuggees, pattern):
        """Finds the debuggee which matches the given pattern.

    Args:
      all_debuggees: A list of debuggees to search.
      pattern: A string containing a debuggee ID or a regular expression that
        matches a single debuggee's name or description. If it matches any
        debuggee name, the description will not be inspected.
    Returns:
      The matching Debuggee.
    Raises:
      errors.MultipleDebuggeesError if the pattern matches multiple debuggees.
      errors.NoDebuggeeError if the pattern matches no debuggees.
    """
        if not all_debuggees:
            raise errors.NoDebuggeeError()
        latest_debuggees = _FilterStaleMinorVersions(all_debuggees)
        debuggees = [d for d in all_debuggees if d.target_id == pattern] + [d for d in latest_debuggees if pattern == d.name]
        if not debuggees:
            match_re = re.compile(pattern)
            debuggees = [d for d in latest_debuggees if match_re.search(d.name)] + [d for d in latest_debuggees if d.description and match_re.search(d.description)]
        if not debuggees:
            raise errors.NoDebuggeeError(pattern, debuggees=all_debuggees)
        debuggee_ids = set((d.target_id for d in debuggees))
        if len(debuggee_ids) > 1:
            raise errors.MultipleDebuggeesError(pattern, debuggees)
        return debuggees[0]

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