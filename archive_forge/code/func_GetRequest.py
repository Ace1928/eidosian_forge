from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import display_helper
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
@staticmethod
def GetRequest(messages, resource, args):
    backend_filter = None
    if args.filter:
        backend_filter = args.filter
        args.filter = None
    return messages.DataprocProjectsLocationsSessionsListRequest(filter=backend_filter, pageSize=args.page_size, parent=resource.RelativeName())