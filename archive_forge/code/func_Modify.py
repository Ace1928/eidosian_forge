from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import health_checks_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.health_checks import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def Modify(self, client, args, existing_check):
    """Returns a modified HealthCheck message."""
    if existing_check.type != client.messages.HealthCheck.TypeValueValuesEnum.UDP:
        raise core_exceptions.Error('update udp subcommand applied to health check with protocol ' + existing_check.type.name)
    if args.description:
        description = args.description
    elif args.description is None:
        description = existing_check.description
    else:
        description = None
    if args.port_name:
        port_name = args.port_name
    elif args.port_name is None:
        port_name = existing_check.udpHealthCheck.portName
    else:
        port_name = None
    new_health_check = client.messages.HealthCheck(name=existing_check.name, description=description, type=client.messages.HealthCheck.TypeValueValuesEnum.UDP, udpHealthCheck=client.messages.UDPHealthCheck(request=args.request or existing_check.udpHealthCheck.request, response=args.response or existing_check.udpHealthCheck.response, port=args.port or existing_check.udpHealthCheck.port, portName=port_name), checkIntervalSec=args.check_interval or existing_check.checkIntervalSec, timeoutSec=args.timeout or existing_check.timeoutSec, healthyThreshold=args.healthy_threshold or existing_check.healthyThreshold, unhealthyThreshold=args.unhealthy_threshold or existing_check.unhealthyThreshold)
    return new_health_check