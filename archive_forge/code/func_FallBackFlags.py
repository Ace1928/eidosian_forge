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
def FallBackFlags(args):
    if not args.organization and (not args.folder) and (not args.project):
        args.organization = properties.VALUES.scc.organization.Get()
        if not args.organization:
            args.project = properties.VALUES.core.project.Get()
    if not args.organization and (not args.folder) and (not args.project):
        raise calliope_exceptions.MinimumArgumentException(['--organization', '--folder', '--project'])