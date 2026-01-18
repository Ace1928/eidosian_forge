from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def SsdCache(positional=False, required=False, hidden=True, text='Cloud Spanner SSD Cache ID.'):
    if positional:
        return base.Argument('cache_id', hidden=hidden, help=text)
    else:
        return base.Argument('--ssd-cache', required=required, hidden=hidden, help=text)