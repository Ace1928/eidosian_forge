from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.source import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddPushblockFlags(group):
    """Add pushblock enabled/disabled flags to the given group."""
    group.add_argument('--enable-pushblock', action='store_true', help='Enable PushBlock for all repositories under current project.\nPushBlock allows repository owners to block git push transactions containing\nprivate key data.')
    group.add_argument('--disable-pushblock', action='store_true', help='Disable PushBlock for all repositories under current project.\nPushBlock allows repository owners to block git push transactions containing\nprivate key data.')