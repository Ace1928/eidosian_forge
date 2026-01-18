from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ConstructInstanceFromArgs(client, alloydb_messages, args):
    """Validates command line input arguments and passes parent's resources to create an AlloyDB instance.

  Args:
    client: Client for api_utils.py class.
    alloydb_messages: Messages module for the API client.
    args: Command line input arguments.

  Returns:
    An AlloyDB instance to create with the specified command line arguments.
  """
    instance_resource = alloydb_messages.Instance()
    instance_resource.availabilityType = ParseAvailabilityType(alloydb_messages, args.availability_type)
    instance_resource.machineConfig = alloydb_messages.MachineConfig(cpuCount=args.cpu_count)
    instance_ref = client.resource_parser.Create('alloydb.projects.locations.clusters.instances', projectsId=properties.VALUES.core.project.GetOrFail, locationsId=args.region, clustersId=args.cluster, instancesId=args.instance)
    instance_resource.name = instance_ref.RelativeName()
    instance_resource.databaseFlags = labels_util.ParseCreateArgs(args, alloydb_messages.Instance.DatabaseFlagsValue, labels_dest='database_flags')
    instance_resource.instanceType = _ParseInstanceType(alloydb_messages, args.instance_type)
    if instance_resource.instanceType == alloydb_messages.Instance.InstanceTypeValueValuesEnum.READ_POOL:
        instance_resource.readPoolConfig = alloydb_messages.ReadPoolConfig(nodeCount=args.read_pool_node_count)
    instance_resource.queryInsightsConfig = _QueryInsightsConfig(alloydb_messages, insights_config_query_string_length=args.insights_config_query_string_length, insights_config_query_plans_per_minute=args.insights_config_query_plans_per_minute, insights_config_record_application_tags=args.insights_config_record_application_tags, insights_config_record_client_address=args.insights_config_record_client_address)
    instance_resource.clientConnectionConfig = _ClientConnectionConfig(alloydb_messages, args.ssl_mode, args.require_connectors)
    instance_resource.networkConfig = _NetworkConfig(alloydb_messages, args.assign_inbound_public_ip, None)
    if args.allowed_psc_projects:
        instance_resource.pscInstanceConfig = _PscInstanceConfig(alloydb_messages, args.allowed_psc_projects)
    return instance_resource