from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddQuicOverrideCreateArgs(parser, default='NONE'):
    """Adds parser arguments for creation related to QuicOverride."""
    parser.add_argument('--quic-override', choices={'NONE': 'Allows Google to control when QUIC is rolled out.', 'ENABLE': 'Allows load balancer to negotiate QUIC with clients.', 'DISABLE': 'Disallows load balancer to negotiate QUIC with clients.'}, default=default, help='Controls whether load balancer may negotiate QUIC with clients. QUIC is a new transport which reduces latency compared to that of TCP. See https://www.chromium.org/quic for more details.')