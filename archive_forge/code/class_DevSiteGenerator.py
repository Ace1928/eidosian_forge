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
class DevSiteGenerator(walker.Walker):
    """Generates DevSite reference HTML in a directory hierarchy.

  This implements gcloud meta generate-help-docs --manpage-dir=DIRECTORY.

  Attributes:
    _directory: The DevSite reference output directory.
    _need_section_tag[]: _need_section_tag[i] is True if there are section
      subitems at depth i. This prevents the creation of empty 'section:' tags
      in the '_toc' files.
    _toc_root: The root TOC output stream.
    _toc_main: The current main (just under root) TOC output stream.
  """
    _REFERENCE = '/sdk/gcloud/reference'
    _TOC = '_toc.yaml'

    def __init__(self, cli, directory, hidden=False, progress_callback=None, restrict=None):
        """Constructor.

    Args:
      cli: The Cloud SDK CLI object.
      directory: The devsite output directory path name.
      hidden: Boolean indicating whether to consider the hidden CLI.
      progress_callback: f(float), The function to call to update the progress
        bar or None for no progress bar.
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups.

    """
        super(DevSiteGenerator, self).__init__(cli)
        self._directory = directory
        files.MakeDir(self._directory)
        self._need_section_tag = []
        toc_path = os.path.join(self._directory, self._TOC)
        self._toc_root = files.FileWriter(toc_path)
        self._toc_root.write('toc:\n')
        self._toc_root.write('- title: "gcloud Reference"\n')
        self._toc_root.write('  path: %s\n' % self._REFERENCE)
        self._toc_root.write('  section:\n')
        self._toc_main = None

    def Visit(self, node, parent, is_group):
        """Updates the TOC and Renders a DevSite doc for each node in the CLI tree.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a group, otherwise its is a command.

    Returns:
      The parent value, ignored here.
    """

        def _UpdateTOC():
            """Updates the DevSIte TOC."""
            depth = len(command) - 1
            if not depth:
                return
            title = ' '.join(command)
            while depth >= len(self._need_section_tag):
                self._need_section_tag.append(False)
            if depth == 1:
                if is_group:
                    if self._toc_main:
                        self._toc_main.close()
                    toc_path = os.path.join(directory, self._TOC)
                    toc = files.FileWriter(toc_path)
                    self._toc_main = toc
                    toc.write('toc:\n')
                    toc.write('- title: "%s"\n' % title)
                    toc.write('  path: %s\n' % '/'.join([self._REFERENCE] + command[1:]))
                    self._need_section_tag[depth] = True
                toc = self._toc_root
                indent = '  '
                if is_group:
                    toc.write('%s- include: %s\n' % (indent, '/'.join([self._REFERENCE] + command[1:] + [self._TOC])))
                    return
            else:
                toc = self._toc_main
                indent = '  ' * (depth - 1)
                if self._need_section_tag[depth - 1]:
                    self._need_section_tag[depth - 1] = False
                    toc.write('%ssection:\n' % indent)
                title = command[-1]
            toc.write('%s- title: "%s"\n' % (indent, title))
            toc.write('%s  path: %s\n' % (indent, '/'.join([self._REFERENCE] + command[1:])))
            self._need_section_tag[depth] = is_group
        command = node.GetPath()
        if is_group:
            directory = os.path.join(self._directory, *command[1:])
            files.MakeDir(directory, mode=493)
        else:
            directory = os.path.join(self._directory, *command[1:-1])
        path = os.path.join(directory, 'index' if is_group else command[-1]) + '.html'
        universe_domain = None
        if properties.VALUES.core.universe_domain.IsExplicitlySet():
            universe_domain = properties.VALUES.core.universe_domain.Get()
        properties.VALUES.core.universe_domain.Set('universe')
        with files.FileWriter(path) as f:
            md = markdown.Markdown(node)
            render_document.RenderDocument(style='devsite', title=' '.join(command), fin=io.StringIO(md), out=f, command_node=node)
        properties.VALUES.core.universe_domain.Set(universe_domain)
        _UpdateTOC()
        return parent

    def Done(self):
        """Closes the TOC files after the CLI tree walk is done."""
        self._toc_root.close()
        if self._toc_main:
            self._toc_main.close()