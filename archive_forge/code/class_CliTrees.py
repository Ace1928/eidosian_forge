from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
class CliTrees(base.Group):
    """CLI trees manager.

  The *{command}* group generates, updates and lists static CLI trees.

  A CLI tree is a module or JSON file that describes a command and its
  subcommands, flags, arguments, help text and TAB completers.
  *$ gcloud interactive* uses CLI trees for typeahead, command line completion,
  and as-you-type documentation, *$ gcloud* uses its CLI tree for static
  completion, and *$ gcloud search help* uses the gcloud CLI tree to search
  help text.

  Packaged CLI tree files are installed in the *cli/data* subdirectory of the
  *gcloud* installation root directory.  These trees are updated by
  *$ gcloud components install* and *$ gcloud components update*. Other CLI
  trees are generated on demand and cached in the per project *cli* config
  directory. Each CLI tree is version-stamped to its command version and
  is updated when the command changes.
  """