from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.calliope import cli_tree_markdown as markdown
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.document_renderers import token_renderer
from prompt_toolkit.layout import controls
def GenerateHelpForFlag(cli, token, width):
    """Returns help lines for a flag token."""
    gen = markdown.CliTreeMarkdownGenerator(cli.root, cli.root)
    gen.PrintFlagDefinition(token.tree)
    mark = gen.Edit()
    fin = io.StringIO(mark)
    return render_document.MarkdownRenderer(token_renderer.TokenRenderer(width=width, height=cli.config.help_lines), fin=fin).Run()