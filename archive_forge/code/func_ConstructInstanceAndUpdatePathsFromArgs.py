from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def ConstructInstanceAndUpdatePathsFromArgs(alloydb_messages, instance_ref, args):
    """Validates command line arguments and creates the instance and update paths.

  Args:
    alloydb_messages: Messages module for the API client.
    instance_ref: parent resource path of the resource being updated
    args: Command line input arguments.

  Returns:
    An AlloyDB instance and paths for update.
  """
    availability_type_path = 'availabilityType'
    database_flags_path = 'databaseFlags'
    cpu_count_path = 'machineConfig.cpuCount'
    read_pool_node_count_path = 'readPoolConfig.nodeCount'
    insights_config_query_string_length_path = 'queryInsightsConfig.queryStringLength'
    insights_config_query_plans_per_minute_path = 'queryInsightsConfig.queryPlansPerMinute'
    insights_config_record_application_tags_path = 'queryInsightsConfig.recordApplicationTags'
    insights_config_record_client_address_path = 'queryInsightsConfig.recordClientAddress'
    instance_resource = alloydb_messages.Instance()
    paths = []
    instance_resource.name = instance_ref.RelativeName()
    availability_type = ParseAvailabilityType(alloydb_messages, args.availability_type)
    if availability_type:
        instance_resource.availabilityType = availability_type
        paths.append(availability_type_path)
    database_flags = labels_util.ParseCreateArgs(args, alloydb_messages.Instance.DatabaseFlagsValue, labels_dest='database_flags')
    if database_flags:
        instance_resource.databaseFlags = database_flags
        paths.append(database_flags_path)
    if args.cpu_count:
        instance_resource.machineConfig = alloydb_messages.MachineConfig(cpuCount=args.cpu_count)
        paths.append(cpu_count_path)
    if args.read_pool_node_count:
        instance_resource.readPoolConfig = alloydb_messages.ReadPoolConfig(nodeCount=args.read_pool_node_count)
        paths.append(read_pool_node_count_path)
    if args.insights_config_query_string_length:
        paths.append(insights_config_query_string_length_path)
    if args.insights_config_query_plans_per_minute:
        paths.append(insights_config_query_plans_per_minute_path)
    if args.insights_config_record_application_tags is not None:
        paths.append(insights_config_record_application_tags_path)
    if args.insights_config_record_client_address is not None:
        paths.append(insights_config_record_client_address_path)
    instance_resource.queryInsightsConfig = _QueryInsightsConfig(alloydb_messages, args.insights_config_query_string_length, args.insights_config_query_plans_per_minute, args.insights_config_record_application_tags, args.insights_config_record_client_address)
    if args.require_connectors is not None:
        require_connectors_path = 'clientConnectionConfig.requireConnectors'
        paths.append(require_connectors_path)
    if args.ssl_mode:
        ssl_mode_path = 'clientConnectionConfig.sslConfig.sslMode'
        paths.append(ssl_mode_path)
    if args.require_connectors is not None or args.ssl_mode:
        instance_resource.clientConnectionConfig = _ClientConnectionConfig(alloydb_messages, args.ssl_mode, args.require_connectors)
    if args.assign_inbound_public_ip or args.authorized_external_networks is not None:
        instance_resource.networkConfig = _NetworkConfig(alloydb_messages, args.assign_inbound_public_ip, args.authorized_external_networks)
    if args.assign_inbound_public_ip and (not instance_resource.networkConfig.enablePublicIp):
        paths.append('networkConfig')
    else:
        if args.assign_inbound_public_ip:
            paths.append('networkConfig.enablePublicIp')
        if args.authorized_external_networks is not None:
            paths.append('networkConfig.authorizedExternalNetworks')
    if args.allowed_psc_projects is not None:
        instance_resource.pscInstanceConfig = _PscInstanceConfig(alloydb_messages, args.allowed_psc_projects)
        paths.append('pscInstanceConfig.allowedConsumerProjects')
    return (instance_resource, paths)