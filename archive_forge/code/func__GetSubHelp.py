from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def _GetSubHelp(self, is_group=False):
    """Returns the help dict indexed by command for sub commands or groups."""
    return {name: usage_text.HelpInfo(help_text=subcommand[cli_tree.LOOKUP_CAPSULE], is_hidden=subcommand.get(cli_tree.LOOKUP_IS_HIDDEN, subcommand.get('hidden', False)), release_track=_GetReleaseTrackFromId(subcommand[cli_tree.LOOKUP_RELEASE])) for name, subcommand in six.iteritems(self._command[cli_tree.LOOKUP_COMMANDS]) if subcommand[cli_tree.LOOKUP_IS_GROUP] == is_group}