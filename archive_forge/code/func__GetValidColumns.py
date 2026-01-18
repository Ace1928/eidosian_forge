from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
def _GetValidColumns(self, args):
    """Returns a list of valid columns."""
    columns = ['name:label=NAME', 'region.basename():label=REGION', 'type:label=PROTOCOL']
    if args.protocol is not None:
        protocol_value = self._ConvertProtocolArgToValue(args)
        if protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.GRPC.number:
            columns.extend(['grpcHealthCheck.port:label=PORT', 'grpcHealthCheck.grpcServiceName:label=GRPC_SERVICE_NAME'])
        elif protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.HTTP.number:
            columns.extend(['httpHealthCheck.host:label=HOST', 'httpHealthCheck.port:label=PORT', 'httpHealthCheck.requestPath:label=REQUEST_PATH', 'httpHealthCheck.proxyHeader:label=PROXY_HEADER'])
        elif protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.HTTPS.number:
            columns.extend(['httpsHealthCheck.host:label=HOST', 'httpsHealthCheck.port:label=PORT', 'httpsHealthCheck.requestPath:label=REQUEST_PATH', 'httpsHealthCheck.proxyHeader:label=PROXY_HEADER'])
        elif protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.HTTP2.number:
            columns.extend(['http2HealthCheck.host:label=HOST', 'http2HealthCheck.port:label=PORT', 'http2HealthCheck.requestPath:label=REQUEST_PATH', 'http2HealthCheck.proxyHeader:label=PROXY_HEADER'])
        elif protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.TCP.number:
            columns.extend(['tcpHealthCheck.port:label=PORT', 'tcpHealthCheck.request:label=REQUEST', 'tcpHealthCheck.response:label=RESPONSE', 'tcpHealthCheck.proxyHeader:label=PROXY_HEADER'])
        elif protocol_value == self.messages.HealthCheck.TypeValueValuesEnum.SSL.number:
            columns.extend(['sslHealthCheck.port:label=PORT', 'sslHealthCheck.request:label=REQUEST', 'sslHealthCheck.response:label=RESPONSE', 'sslHealthCheck.proxyHeader:label=PROXY_HEADER'])
    return columns