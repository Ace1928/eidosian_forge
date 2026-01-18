from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
class ZonalParallelLister(object):
    """List zonal resources from all zones in parallel (in one batch).

  This class can be used to list only zonal resources.

  This class should not be inherited.

  Attributes:
    client: The compute client.
    service: Zonal service whose resources will be listed.
    resources: The compute resource registry.
    allow_partial_server_failure: Allows Lister to continue presenting items
      from scopes that return succesfully while logging failures as a warning.
  """

    def __init__(self, client, service, resources, allow_partial_server_failure=True):
        self.client = client
        self.service = service
        self.resources = resources
        self.allow_partial_server_failure = allow_partial_server_failure

    def __deepcopy__(self, memodict=None):
        return self

    def __eq__(self, other):
        if not isinstance(other, ZonalParallelLister):
            return False
        return self.client == other.client and self.service == other.service

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.client, self.service))

    def __repr__(self):
        return 'ZonalParallelLister({}, {}, {})'.format(repr(self.client), repr(self.service), repr(self.resources))

    def __call__(self, frontend):
        scope_set = frontend.scope_set
        filter_expr = frontend.filter
        if isinstance(scope_set, ZoneSet):
            zones = scope_set
        else:
            zones_list_data = _Frontend(scopeSet=GlobalScope(scope_set.projects))
            zones_list_implementation = MultiScopeLister(self.client, global_service=self.client.apitools_client.zones)
            zones = ZoneSet([self.resources.Parse(z['selfLink']) for z in Invoke(zones_list_data, zones_list_implementation)])
        service_list_data = _Frontend(filter_expr=filter_expr, maxResults=frontend.max_results, scopeSet=zones)
        service_list_implementation = MultiScopeLister(self.client, zonal_service=self.service, allow_partial_server_failure=self.allow_partial_server_failure)
        return Invoke(service_list_data, service_list_implementation)