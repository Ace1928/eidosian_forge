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
class DatabaseRoleCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(DatabaseRoleCompleter, self).__init__(collection='spanner.projects.instances.databases.roles', list_command='beta spanner databases roles list --uri', flags=['database', 'instance'], **kwargs)