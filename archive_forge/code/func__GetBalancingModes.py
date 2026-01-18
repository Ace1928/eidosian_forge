from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def _GetBalancingModes():
    """Returns the --balancing-modes flag value choices name:description dict."""
    per_rate_flags = '*--max-rate-per-instance*'
    per_connection_flags = '*--max-connections-per-instance*'
    per_rate_flags += '/*--max-rate-per-endpoint*'
    per_connection_flags += '*--max-max-per-endpoint*'
    utilization_extra_help = 'This is incompatible with --network-endpoint-group.'
    balancing_modes = {'CONNECTION': textwrap.dedent("\n          Available if the backend service's load balancing scheme is either\n          `INTERNAL` or `EXTERNAL`.\n          Available if the backend service's protocol is one of `SSL`, `TCP`,\n          or `UDP`.\n\n          Spreads load based on how many concurrent connections the backend\n          can handle.\n\n          For backend services with --load-balancing-scheme `EXTERNAL`, you\n          must specify exactly one of these additional parameters:\n          `--max-connections`, `--max-connections-per-instance`, or\n          `--max-connections-per-endpoint`.\n\n          For backend services where `--load-balancing-scheme` is `INTERNAL`,\n          you must omit all of these parameters.\n          ").format(per_rate_flags), 'RATE': textwrap.dedent("\n          Available if the backend service's load balancing scheme is\n          `INTERNAL_MANAGED`, `INTERNAL_SELF_MANAGED`, or `EXTERNAL`. Available\n          if the backend service's protocol is one of HTTP, HTTPS, or HTTP/2.\n\n          Spreads load based on how many HTTP requests per second (RPS) the\n          backend can handle.\n\n          You must specify exactly one of these additional parameters:\n          `--max-rate`, `--max-rate-per-instance`, or `--max-rate-per-endpoint`.\n          ").format(utilization_extra_help), 'UTILIZATION': textwrap.dedent("\n          Available if the backend service's load balancing scheme is\n          `INTERNAL_MANAGED`, `INTERNAL_SELF_MANAGED`, or `EXTERNAL`. Available only\n          for managed or unmanaged instance group backends.\n\n          Spreads load based on the backend utilization of instances in a backend\n          instance group.\n\n          The following additional parameters may be specified:\n          `--max-utilization`, `--max-rate`, `--max-rate-per-instance`,\n          `--max-connections`, `--max-connections-per-instance`.\n          For valid combinations, see `--max-utilization`.\n          ").format(per_connection_flags)}
    return balancing_modes