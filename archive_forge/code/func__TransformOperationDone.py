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
def _TransformOperationDone(resource):
    """Combines done and throttled fields into a single column."""
    done_cell = '{0}'.format(resource.get('done', False))
    if resource.get('metadata', {}).get('throttled', False):
        done_cell += ' (throttled)'
    return done_cell