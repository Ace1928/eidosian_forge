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
def Database(positional=True, required=True, text='Cloud Spanner database ID.'):
    if positional:
        return base.Argument('database', completer=DatabaseCompleter, help=text)
    else:
        return base.Argument('--database', required=required, completer=DatabaseCompleter, help=text)