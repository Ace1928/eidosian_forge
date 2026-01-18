from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetProjectFlag(verb):
    return base.Argument('id', metavar='PROJECT_ID', completer=completers.ProjectCompleter, help='ID for the project you want to {0}.'.format(verb))