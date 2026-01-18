from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import health_checks_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.health_checks import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def _Modify(client, args, existing_check, include_log_config, include_weighted_load_balancing, include_source_regions):
    """Returns a modified HealthCheck message."""
    if existing_check.type != client.messages.HealthCheck.TypeValueValuesEnum.HTTPS:
        raise core_exceptions.Error('update https subcommand applied to health check with protocol ' + existing_check.type.name)
    if args.description:
        description = args.description
    elif args.description is None:
        description = existing_check.description
    else:
        description = None
    if args.host:
        host = args.host
    elif args.host is None:
        host = existing_check.httpsHealthCheck.host
    else:
        host = None
    port, port_name, port_specification = health_checks_utils.HandlePortRelatedFlagsForUpdate(args, existing_check.httpsHealthCheck)
    if include_weighted_load_balancing:
        weight_report_mode = existing_check.httpsHealthCheck.weightReportMode
        if args.IsSpecified('weight_report_mode'):
            weight_report_mode = client.messages.HTTPSHealthCheck.WeightReportModeValueValuesEnum(args.weight_report_mode)
    proxy_header = existing_check.httpsHealthCheck.proxyHeader
    if args.proxy_header is not None:
        proxy_header = client.messages.HTTPSHealthCheck.ProxyHeaderValueValuesEnum(args.proxy_header)
    if args.response:
        response = args.response
    elif args.response is None:
        response = existing_check.httpsHealthCheck.response
    else:
        response = None
    https_health_check = client.messages.HTTPSHealthCheck(host=host, port=port, portName=port_name, requestPath=args.request_path or existing_check.httpsHealthCheck.requestPath, portSpecification=port_specification, proxyHeader=proxy_header, response=response)
    if include_weighted_load_balancing:
        https_health_check.weightReportMode = weight_report_mode
    new_health_check = client.messages.HealthCheck(name=existing_check.name, description=description, type=client.messages.HealthCheck.TypeValueValuesEnum.HTTPS, httpsHealthCheck=https_health_check, checkIntervalSec=args.check_interval or existing_check.checkIntervalSec, timeoutSec=args.timeout or existing_check.timeoutSec, healthyThreshold=args.healthy_threshold or existing_check.healthyThreshold, unhealthyThreshold=args.unhealthy_threshold or existing_check.unhealthyThreshold)
    if include_log_config:
        new_health_check.logConfig = health_checks_utils.ModifyLogConfig(client, args, existing_check.logConfig)
    if include_source_regions:
        source_regions = existing_check.sourceRegions
        if args.IsSpecified('source_regions'):
            source_regions = args.source_regions
        new_health_check.sourceRegions = source_regions
    return new_health_check