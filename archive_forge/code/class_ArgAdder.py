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
class ArgAdder(object):
    """A class for adding Bigtable command-line arguments."""

    def __init__(self, parser):
        self.parser = parser

    def AddAsync(self):
        base.ASYNC_FLAG.AddToParser(self.parser)
        return self

    def AddCluster(self):
        """Add cluster argument."""
        self.parser.add_argument('--cluster', completer=ClusterCompleter, help='ID of the cluster.', required=True)
        return self

    def AddDeprecatedCluster(self):
        """Add deprecated cluster argument."""
        self.parser.add_argument('--cluster', completer=ClusterCompleter, help='ID of the cluster', required=False, action=actions.DeprecationAction('--cluster', warn='The {flag_name} argument is deprecated; use --cluster-config instead.', removed=False, action='store'))
        return self

    def AddDeprecatedClusterNodes(self):
        """Add deprecated cluster nodes argument."""
        self.parser.add_argument('--cluster-num-nodes', help='Number of nodes to serve.', required=False, type=int, action=actions.DeprecationAction('--cluster-num-nodes', warn='The {flag_name} argument is deprecated; use --cluster-config instead.', removed=False, action='store'))
        return self

    def AddClusterStorage(self):
        storage_argument = base.ChoiceArgument('--cluster-storage-type', choices=['hdd', 'ssd'], default='ssd', help_str='Storage class for the cluster.')
        storage_argument.AddToParser(self.parser)
        return self

    def AddClusterZone(self, in_instance=False):
        self.parser.add_argument('--cluster-zone' if in_instance else '--zone', help='ID of the zone where the cluster is located. Supported zones are listed at https://cloud.google.com/bigtable/docs/locations.', required=True)
        return self

    def AddDeprecatedClusterZone(self):
        """Add deprecated cluster zone argument."""
        self.parser.add_argument('--cluster-zone', help='ID of the zone where the cluster is located. Supported zones are listed at https://cloud.google.com/bigtable/docs/locations.', required=False, action=actions.DeprecationAction('--cluster-zone', warn='The {flag_name} argument is deprecated; use --cluster-config instead.', removed=False, action='store'))
        return self

    def AddInstance(self, positional=True, required=True, multiple=False, additional_help=None):
        """Add argument for instance ID to parser."""
        help_text = 'ID of the {}.'.format(text.Pluralize(2 if multiple else 1, 'instance'))
        if additional_help:
            help_text = ' '.join([help_text, additional_help])
        name = 'instance' if positional else '--instance'
        args = {'completer': InstanceCompleter, 'help': help_text}
        if multiple:
            if positional:
                args['nargs'] = '+'
            else:
                name = '--instances'
                args['type'] = arg_parsers.ArgList()
                args['metavar'] = 'INSTANCE'
        if not positional:
            args['required'] = required
        self.parser.add_argument(name, **args)
        return self

    def AddTable(self):
        """Add table argument."""
        self.parser.add_argument('--table', completer=TableCompleter, help='ID of the table.', required=True)
        return self

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

    def AddDescription(self, resource, required=True):
        """Add argument for description to parser."""
        self.parser.add_argument('--description', help='Friendly name of the {}.'.format(resource), required=required)
        return self

    def AddForce(self, verb):
        """Add argument for force to the parser."""
        self.parser.add_argument('--force', action='store_true', default=False, help='Ignore warnings and force {}.'.format(verb))
        return self

    def AddIsolation(self, allow_data_boost=False):
        """Add argument for isolating this app profile's traffic to parser."""
        isolation_group = self.parser.add_mutually_exclusive_group()
        standard_isolation_group = isolation_group.add_group('Standard Isolation')
        choices = {'PRIORITY_LOW': 'Requests are treated with low priority.', 'PRIORITY_MEDIUM': 'Requests are treated with medium priority.', 'PRIORITY_HIGH': 'Requests are treated with high priority.'}
        standard_isolation_group.add_argument('--priority', type=lambda x: x.replace('-', '_').upper(), choices=choices, default=None, help='Specify the request priority under Standard Isolation. Passing this option implies Standard Isolation, e.g. the `--standard` option. If not specified, the app profile uses Standard Isolation with PRIORITY_HIGH by default. Specifying request priority on an app profile that has Data Boost Read-Only Isolation enabled will change the isolation to Standard and use the specified priority, which may cause unexpected behavior for running applications.' if allow_data_boost else 'Specify the request priority. If not specified, the app profile uses PRIORITY_HIGH by default.', required=True)
        if allow_data_boost:
            standard_isolation_group.add_argument('--standard', action='store_true', default=False, help='Use Standard Isolation, rather than Data Boost Read-only Isolation. If specified, `--priority` is required.')
            data_boost_isolation_group = isolation_group.add_group('Data Boost Read-only Isolation')
            data_boost_isolation_group.add_argument('--data-boost', action='store_true', default=False, help='Use Data Boost Read-only Isolation, rather than Standard Isolation. If specified, --data-boost-compute-billing-owner is required. Specifying Data Boost Read-only Isolation on an app profile which has Standard Isolation enabled may cause unexpected behavior for running applications.', required=True)
            compute_billing_choices = {'HOST_PAYS': 'Compute Billing should be accounted towards the host Cloud Project (containing the targeted Bigtable Instance / Table).'}
            data_boost_isolation_group.add_argument('--data-boost-compute-billing-owner', type=lambda x: x.upper(), choices=compute_billing_choices, default=None, help='Specify the Data Boost Compute Billing Owner, required if --data-boost is passed.', required=True)
        return self

    def AddInstanceDisplayName(self, required=False):
        """Add argument group for display-name to parser."""
        self.parser.add_argument('--display-name', help='Friendly name of the instance.', required=required)
        return self

    def AddDeprecatedInstanceType(self):
        """Add deprecated instance type argument."""
        choices = {'PRODUCTION': 'Production instances provide high availability and are suitable for applications in production. Production instances created with the --instance-type argument have 3 nodes if a value is not provided for --cluster-num-nodes.', 'DEVELOPMENT': 'Development instances are low-cost instances meant for development and testing only. They do not provide high availability and no service level agreement applies.'}
        self.parser.add_argument('--instance-type', default='PRODUCTION', type=lambda x: x.upper(), choices=choices, help='The type of instance to create.', required=False, action=actions.DeprecationAction('--instance-type', warn='The {flag_name} argument is deprecated. DEVELOPMENT instances are no longer offered. All instances are of type PRODUCTION.', removed=False, action='store'))
        return self

    def AddClusterConfig(self):
        """Add the cluster-config argument as repeated kv dicts."""
        self.parser.add_argument('--cluster-config', action='append', type=arg_parsers.ArgDict(spec={'id': str, 'zone': str, 'nodes': int, 'kms-key': str, 'autoscaling-min-nodes': int, 'autoscaling-max-nodes': int, 'autoscaling-cpu-target': int, 'autoscaling-storage-target': int}, required_keys=['id', 'zone'], max_length=8), metavar='id=ID,zone=ZONE,nodes=NODES,kms-key=KMS_KEY,autoscaling-min-nodes=AUTOSCALING_MIN_NODES,autoscaling-max-nodes=AUTOSCALING_MAX_NODES,autoscaling-cpu-target=AUTOSCALING_CPU_TARGET,autoscaling-storage-target=AUTOSCALING_STORAGE_TARGET', help=textwrap.dedent('        *Repeatable*. Specify cluster config as a key-value dictionary.\n\n        This is the recommended argument for specifying cluster configurations.\n\n        Keys can be:\n\n          *id*: Required. The ID of the cluster.\n\n          *zone*: Required. ID of the zone where the cluster is located. Supported zones are listed at https://cloud.google.com/bigtable/docs/locations.\n\n          *nodes*: The number of nodes in the cluster. Default=1.\n\n          *kms-key*: The Cloud KMS (Key Management Service) cryptokey that will be used to protect the cluster.\n\n          *autoscaling-min-nodes*: The minimum number of nodes for autoscaling.\n\n          *autoscaling-max-nodes*: The maximum number of nodes for autoscaling.\n\n          *autoscaling-cpu-target*: The target CPU utilization percentage for autoscaling. Accepted values are from 10 to 80.\n\n          *autoscaling-storage-target*: The target storage utilization gibibytes per node for autoscaling. Accepted values are from 2560 to 5120 for SSD clusters and 8192 to 16384 for HDD clusters.\n\n        If this argument is specified, the deprecated arguments for configuring a single cluster will be ignored, including *--cluster*, *--cluster-zone*, *--cluster-num-nodes*.\n\n        See *EXAMPLES* section.\n        '))
        return self

    def AddScalingArgs(self, required=False, num_nodes_required=False, num_nodes_default=None, add_disable_autoscaling=False, require_all_essential_autoscaling_args=False):
        """Add scaling related arguments."""
        scaling_group = self.parser.add_mutually_exclusive_group(required=required)
        manual_scaling_group = scaling_group.add_group('Manual Scaling')
        manual_scaling_group.add_argument('--num-nodes', help='Number of nodes to serve.', default=num_nodes_default, required=num_nodes_required, type=int, metavar='NUM_NODES')
        if add_disable_autoscaling:
            manual_scaling_group.add_argument('--disable-autoscaling', help='Set this flag and --num-nodes to disable autoscaling. If autoscaling is currently not enabled, setting this flag does nothing.', action='store_true', default=False, required=False, hidden=False)
        autoscaling_group = scaling_group.add_group('Autoscaling', hidden=False)
        autoscaling_group.add_argument('--autoscaling-min-nodes', help='The minimum number of nodes for autoscaling.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_MIN_NODES')
        autoscaling_group.add_argument('--autoscaling-max-nodes', help='The maximum number of nodes for autoscaling.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_MAX_NODES')
        autoscaling_group.add_argument('--autoscaling-cpu-target', help='The target CPU utilization percentage for autoscaling. Accepted values are from 10 to 80.', default=None, required=require_all_essential_autoscaling_args, type=int, metavar='AUTOSCALING_CPU_TARGET')
        autoscaling_group.add_argument('--autoscaling-storage-target', help='The target storage utilization gibibytes per node for autoscaling. Accepted values are from 2560 to 5120 for SSD clusters and 8192 to 16384 for HDD clusters.', default=None, required=False, type=int, metavar='AUTOSCALING_STORAGE_TARGET')
        return self

    def AddScalingArgsForClusterUpdate(self):
        """Add scaling related arguments."""
        return self.AddScalingArgs(required=True, num_nodes_required=True, add_disable_autoscaling=True)

    def AddScalingArgsForClusterCreate(self):
        """Add scaling related arguments."""
        return self.AddScalingArgs(num_nodes_default=3, require_all_essential_autoscaling_args=True)