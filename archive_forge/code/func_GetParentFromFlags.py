from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import six
def GetParentFromFlags(args):
    if getattr(args, 'folder', None):
        return 'folders/{0}'.format(args.folder)
    elif args.organization:
        return 'organizations/{0}'.format(args.organization)
    else:
        return None