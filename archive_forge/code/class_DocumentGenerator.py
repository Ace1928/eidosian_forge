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
class DocumentGenerator(walker.Walker):
    """Generates style manpage files with suffix in an output directory.

  All files will be generated in one directory.

  Attributes:
    _directory: The document output directory.
    _style: The document style.
    _suffix: The output file suffix.
  """

    def __init__(self, cli, directory, style, suffix):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      directory: The manpage output directory path name.
      style: The document style.
      suffix: The generate document file suffix. None for .<SECTION>.
    """
        super(DocumentGenerator, self).__init__(cli)
        self._directory = directory
        self._style = style
        self._suffix = suffix
        files.MakeDir(self._directory)

    def Visit(self, node, parent, is_group):
        """Renders document file for each node in the CLI tree.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a group, otherwise its is a command.

    Returns:
      The parent value, ignored here.
    """
        if self._style == 'linter':
            meta_data = actions.GetCommandMetaData(node)
        else:
            meta_data = None
        command = node.GetPath()
        path = os.path.join(self._directory, '_'.join(command)) + self._suffix
        with files.FileWriter(path) as f:
            md = markdown.Markdown(node)
            render_document.RenderDocument(style=self._style, title=' '.join(command), fin=io.StringIO(md), out=f, command_metadata=meta_data)
        return parent