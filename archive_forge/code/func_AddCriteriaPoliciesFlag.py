from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddCriteriaPoliciesFlag(parser, resource):
    parser.add_argument('--criteria-policies', metavar='CRITERIA_POLICIES', type=arg_parsers.ArgList(min_length=1, max_length=16), help='The policies that the {} applies to.'.format(resource))