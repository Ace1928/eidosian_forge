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
def WriteHtmlMenu(self, command, out):
    """Writes the command menu tree HTML on out.

    Args:
      command: dict, The tree (nested dict) of command/group names.
      out: stream, The output stream.
    """

    def ConvertPathToIdentifier(path):
        return '_'.join(path)

    def WalkCommandTree(command, prefix):
        """Visit each command and group in the CLI command tree.

      Args:
        command: dict, The tree (nested dict) of command/group names.
        prefix: [str], The subcommand arg prefix.
      """
        level = len(prefix)
        visibility = 'visible' if level <= 1 else 'hidden'
        indent = level * 2 + 2
        name = command.get('_name_')
        args = prefix + [name]
        out.write('{indent}<li class="{visibility}" id="{item}" onclick="select(event, this.id)">{name}'.format(indent=' ' * indent, visibility=visibility, name=name, item=ConvertPathToIdentifier(args)))
        commands = command.get('commands', []) + command.get('groups', [])
        if commands:
            out.write('<ul>\n')
            for c in sorted(commands, key=lambda x: x['_name_']):
                WalkCommandTree(c, args)
            out.write('{indent}</ul>\n'.format(indent=' ' * (indent + 1)))
            out.write('{indent}</li>\n'.format(indent=' ' * indent))
        else:
            out.write('</li>\n'.format(indent=' ' * (indent + 1)))
    out.write('<html>\n<head>\n<meta name="description" content="man page tree navigation">\n<meta name="generator" content="gcloud meta generate-help-docs --html-dir=.">\n<title> man page tree navigation </title>\n<base href="." target="_blank">\n<link rel="stylesheet" type="text/css" href="_menu_.css">\n<script type="text/javascript" src="_menu_.js"></script>\n</head>\n<body>\n\n<div class="menu">\n <ul>\n')
    WalkCommandTree(command, [])
    out.write(' </ul>\n</div>\n\n</body>\n</html>\n')