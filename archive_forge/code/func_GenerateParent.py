from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.scc.settings import exceptions as scc_exceptions
from googlecloudsdk.core import properties
def GenerateParent(args):
    if args.organization:
        return 'organizations/{}/'.format(args.organization)
    elif args.project:
        return 'projects/{}/'.format(args.project)
    elif args.folder:
        return 'folders/{}/'.format(args.folder)