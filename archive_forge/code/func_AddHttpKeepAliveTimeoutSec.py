from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddHttpKeepAliveTimeoutSec(parser):
    """Adds the http keep alive timeout sec argument."""
    parser.add_argument('--http-keep-alive-timeout-sec', type=arg_parsers.BoundedInt(5, 1200), help='      Represents the maximum amount of time that a TCP connection can be idle\n      between the (downstream) client and the target HTTP proxy. If an HTTP\n      keepalive  timeout is not specified, the default value is 610 seconds.\n      For global external Application Load Balancers, the minimum allowed\n      value is 5 seconds and the maximum allowed value is 1200 seconds.\n      ')