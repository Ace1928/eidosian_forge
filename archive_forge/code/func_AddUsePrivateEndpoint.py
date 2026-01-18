from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddUsePrivateEndpoint(parser):
    """Adds --use-private-endpoint flag."""
    parser.add_argument('--use-private-endpoint', action='store_true', help="Only allow access to the master's private endpoint IP.")