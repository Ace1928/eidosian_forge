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
def _GenerateHtmlNav(self, directory, cli, hidden, restrict):
    """Generates html nav files in directory."""
    tree = CommandTreeGenerator(cli).Walk(hidden, restrict)
    with files.FileWriter(os.path.join(directory, '_menu_.html')) as out:
        self.WriteHtmlMenu(tree, out)
    for file_name in _HELP_HTML_DATA_FILES:
        file_contents = pkg_resources.GetResource('googlecloudsdk.api_lib.meta.help_html_data.', file_name)
        files.WriteBinaryFileContents(os.path.join(directory, file_name), file_contents)