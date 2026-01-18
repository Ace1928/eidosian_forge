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
def ListBreakpoints(self, location_regexp=None, resource_ids=None, include_all_users=False, include_inactive=False, restrict_to_type=None, full_details=False):
    """Returns all breakpoints matching the given IDs or patterns.

    Lists all breakpoints for this debuggee, and returns every breakpoint
    where the location field contains the given pattern or the ID is exactly
    equal to the pattern (there can be at most one breakpoint matching by ID).

    Args:
      location_regexp: A list of regular expressions to compare against the
        location ('path:line') of the breakpoints. If both location_regexp and
        resource_ids are empty or None, all breakpoints will be returned.
      resource_ids: Zero or more resource IDs in the form expected by the
        resource parser. These breakpoints will be retrieved regardless
        of the include_all_users or include_inactive flags
      include_all_users: If true, search breakpoints created by all users.
      include_inactive: If true, search breakpoints that are in the final state.
        This option controls whether regular expressions can match inactive
        breakpoints. If an object is specified by ID, it will be returned
        whether or not this flag is set.
      restrict_to_type: An optional breakpoint type (LOGPOINT_TYPE or
        SNAPSHOT_TYPE)
      full_details: If true, issue a GetBreakpoint request for every result to
        get full details including the call stack and variable table.
    Returns:
      A list of all matching breakpoints.
    Raises:
      InvalidLocationException if a regular expression is not valid.
    """
    resource_ids = resource_ids or []
    location_regexp = location_regexp or []
    ids = set([self._resource_parser.Parse(r, params={'debuggeeId': self.target_id}, collection='clouddebugger.debugger.debuggees.breakpoints').Name() for r in resource_ids])
    patterns = []
    for r in location_regexp:
        try:
            patterns.append(re.compile('^(.*/)?(' + r + ')$'))
        except re.error as e:
            raise errors.InvalidLocationException('The location pattern "{0}" is not a valid Python regular expression: {1}'.format(r, e))
    request = self._debug_messages.ClouddebuggerDebuggerDebuggeesBreakpointsListRequest(debuggeeId=self.target_id, includeAllUsers=include_all_users, includeInactive=include_inactive or bool(ids), clientVersion=self.CLIENT_VERSION)
    try:
        response = self._debug_client.debugger_debuggees_breakpoints.List(request)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)
    if not patterns and (not ids):
        return self._FilteredDictListWithInfo(response.breakpoints, restrict_to_type)
    if include_inactive:
        result = [bp for bp in response.breakpoints if _BreakpointMatchesIdOrRegexp(bp, ids, patterns)]
    else:
        result = [bp for bp in response.breakpoints if _BreakpointMatchesIdOrRegexp(bp, ids, [] if bp.isFinalState else patterns)]
    missing_ids = ids - set([bp.id for bp in result])
    if missing_ids:
        raise errors.BreakpointNotFoundError(missing_ids, self._BreakpointDescription(restrict_to_type))
    for p in patterns:
        if not [bp for bp in result if _BreakpointMatchesIdOrRegexp(bp, [], [p])]:
            raise errors.NoMatchError(self._BreakpointDescription(restrict_to_type), p.pattern)
    result = self._FilteredDictListWithInfo(result, restrict_to_type)
    if full_details:

        def IsCompletedSnapshot(bp):
            return (not bp.action or bp.action == self.BreakpointAction(self.SNAPSHOT_TYPE)) and bp.isFinalState and (not (bp.status and bp.status.isError))
        result = [self.GetBreakpoint(bp.id) if IsCompletedSnapshot(bp) else bp for bp in result]
    return result