from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
def AddAppProfileRouting(self, required=True, allow_failover_radius=False, allow_row_affinity=False):
    """Adds arguments for app_profile routing to parser."""
    routing_group = self.parser.add_mutually_exclusive_group(required=required)
    any_group = routing_group.add_group('Multi Cluster Routing Policy')
    any_group.add_argument('--route-any', action='store_true', required=True, default=False, help='Use Multi Cluster Routing policy.')
    any_group.add_argument('--restrict-to', type=arg_parsers.ArgList(), help='Cluster IDs to route to using the Multi Cluster Routing Policy. If unset, all clusters in the instance are eligible.', metavar='RESTRICT_TO')
    if allow_row_affinity:
        any_group.add_argument('--row-affinity', action='store_true', default=None, help='Use row affinity sticky routing for this app profile.', hidden=True)
    if allow_failover_radius:
        choices = {'ANY_REGION': 'Requests will be allowed to fail over to all eligible clusters.', 'INITIAL_REGION_ONLY': 'Requests will only be allowed to fail over to clusters within the region the request was first routed to.'}
        any_group.add_argument('--failover-radius', type=lambda x: x.replace('-', '_').upper(), choices=choices, help='Restricts clusters that requests can fail over to by proximity. Failover radius must be either any-region or initial-region-only. any-region allows requests to fail over without restriction. initial-region-only prohibits requests from failing over to any clusters outside of the initial region the request was routed to. If omitted, any-region will be used by default.', metavar='FAILOVER_RADIUS', hidden=True)
    route_to_group = routing_group.add_group('Single Cluster Routing Policy')
    route_to_group.add_argument('--route-to', completer=ClusterCompleter, required=True, help='Cluster ID to route to using Single Cluster Routing policy.')
    transactional_write_help = 'Allow transactional writes with a Single Cluster Routing policy.'
    route_to_group.add_argument('--transactional-writes', action='store_true', default=None, help=transactional_write_help)
    return self