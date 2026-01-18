from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddIdArgToParser(parser):
    base.Argument('id', metavar='ORG_POLICY_ID', help='The Org Policy constraint name.').AddToParser(parser)