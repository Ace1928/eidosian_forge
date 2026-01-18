from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class CommandTreeGenerator(walker.Walker):
    """Constructs a CLI command dict tree.

  This implements the resource generator for gcloud meta list-commands.

  Attributes:
    _with_flags: Include the non-global flags for each command/group if True.
    _with_flag_values: Include flag value choices or :type: if True.
    _global_flags: The set of global flags, only listed for the root command.
  """

    def __init__(self, cli, with_flags=False, with_flag_values=False, **kwargs):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      with_flags: Include the non-global flags for each command/group if True.
      with_flag_values: Include flags and flag value choices or :type: if True.
      **kwargs: Other keyword arguments to pass to Walker constructor.
    """
        super(CommandTreeGenerator, self).__init__(cli, **kwargs)
        self._with_flags = with_flags or with_flag_values
        self._with_flag_values = with_flag_values
        self._global_flags = set()

    def Visit(self, node, parent, is_group):
        """Visits each node in the CLI command tree to construct the dict tree.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a group, otherwise its is a command.

    Returns:
      The subtree parent value, used here to construct a dict tree.
    """
        name = node.name.replace('_', '-')
        info = {'_name_': name}
        if self._with_flags:
            all_flags = []
            for arg in node.GetAllAvailableFlags():
                value = None
                if self._with_flag_values:
                    if arg.choices:
                        choices = sorted(arg.choices)
                        if choices != ['false', 'true']:
                            value = ','.join([six.text_type(choice) for choice in choices])
                    elif isinstance(arg.type, int):
                        value = ':int:'
                    elif isinstance(arg.type, float):
                        value = ':float:'
                    elif isinstance(arg.type, arg_parsers.ArgDict):
                        value = ':dict:'
                    elif isinstance(arg.type, arg_parsers.ArgList):
                        value = ':list:'
                    elif arg.nargs != 0:
                        metavar = arg.metavar or arg.dest.upper()
                        value = ':' + metavar + ':'
                for f in arg.option_strings:
                    if value:
                        f += '=' + value
                    all_flags.append(f)
            no_prefix = '--no-'
            flags = []
            for flag in all_flags:
                if flag in self._global_flags:
                    continue
                if flag.startswith(no_prefix):
                    positive = '--' + flag[len(no_prefix):]
                    if positive in all_flags:
                        continue
                flags.append(flag)
            if flags:
                info['_flags_'] = sorted(flags)
                if not self._global_flags:
                    self._global_flags.update(flags)
        if is_group:
            if parent:
                if cli_tree.LOOKUP_GROUPS not in parent:
                    parent[cli_tree.LOOKUP_GROUPS] = []
                parent[cli_tree.LOOKUP_GROUPS].append(info)
            return info
        if cli_tree.LOOKUP_COMMANDS not in parent:
            parent[cli_tree.LOOKUP_COMMANDS] = []
        parent[cli_tree.LOOKUP_COMMANDS].append(info)
        return None