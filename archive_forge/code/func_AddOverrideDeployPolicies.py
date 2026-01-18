from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddOverrideDeployPolicies(parser, hidden=True):
    """Adds override-deploy-policies flag."""
    parser.add_argument('--override-deploy-policies', metavar='POLICY', hidden=hidden, type=arg_parsers.ArgList(), help='Deploy policies to override')