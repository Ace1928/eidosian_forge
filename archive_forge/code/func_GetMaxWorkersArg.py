from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetMaxWorkersArg(required=False):
    return base.Argument('--max-workers', required=required, default=None, type=int, help='Maximum number of workers to run by default. Must be between 1 and 1000.')