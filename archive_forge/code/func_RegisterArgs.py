from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.declarative import flags as declarative_config_flags
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
def RegisterArgs(parser, add_to_parser, **kwargs):
    mutex_group = parser.add_group(mutex=True, required=True)
    resource_group = mutex_group.add_group()
    add_to_parser(resource_group, **kwargs)
    declarative_config_flags.AddAllFlag(mutex_group, collection='project')
    declarative_config_flags.AddPathFlag(parser)
    declarative_config_flags.AddFormatFlag(parser)