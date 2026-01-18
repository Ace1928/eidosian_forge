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
class ManPageGenerator(DocumentGenerator):
    """Generates manpage files with suffix .1 in an output directory.

  The output directory will contain a man1 subdirectory containing all of the
  manpage files.
  """
    _SECTION_FORMAT = 'man{section}'

    def __init__(self, cli, directory, hidden=False, progress_callback=None, restrict=None):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      directory: The manpage output directory path name.
      hidden: Boolean indicating whether to consider the hidden CLI.
      progress_callback: f(float), The function to call to update the progress
        bar or None for no progress bar.
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups.

    """
        section_subdir = self._SECTION_FORMAT.format(section=1)
        section_dir = os.path.join(directory, section_subdir)
        super(ManPageGenerator, self).__init__(cli, directory=section_dir, style='man', suffix='.1')