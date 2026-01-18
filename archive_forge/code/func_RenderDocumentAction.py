from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def RenderDocumentAction(command, default_style=None):
    """Get an argparse.Action that renders a help document from markdown.

  Args:
    command: calliope._CommandCommon, The command object that we're helping.
    default_style: str, The default style if not specified in flag value.

  Returns:
    argparse.Action, The action to use.
  """

    class Action(argparse.Action):
        """The action created for RenderDocumentAction."""

        def __init__(self, **kwargs):
            if default_style:
                kwargs['nargs'] = 0
            super(Action, self).__init__(**kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            """Render a help document according to the style in values.

      Args:
        parser: The ArgParse object.
        namespace: The ArgParse namespace.
        values: The --document flag ArgDict() value:
          style=STYLE
            The output style. Must be specified.
          title=DOCUMENT TITLE
            The document title.
          notes=SENTENCES
            Inserts SENTENCES into the document NOTES section.
        option_string: The ArgParse flag string.

      Raises:
        parser_errors.ArgumentError: For unknown flag value attribute name.
      """
            from googlecloudsdk.calliope import markdown
            from googlecloudsdk.core.document_renderers import render_document
            base.LogCommand(parser.prog, namespace)
            if default_style:
                metrics.Loaded()
            style = default_style
            notes = None
            title = None
            for attributes in values:
                for name, value in six.iteritems(attributes):
                    if name == 'notes':
                        notes = value
                    elif name == 'style':
                        style = value
                    elif name == 'title':
                        title = value
                    else:
                        raise parser_errors.ArgumentError('Unknown document attribute [{0}]'.format(name))
            if title is None:
                title = command.dotted_name
            metrics.Help(command.dotted_name, style)
            if style in ('--help', 'help', 'topic'):
                style = 'text'
            md = io.StringIO(markdown.Markdown(command))
            out = io.StringIO() if console_io.IsInteractive(output=True) else None
            if style == 'linter':
                meta_data = GetCommandMetaData(command)
            else:
                meta_data = None
            if style == 'devsite':
                command_node = command
            else:
                command_node = None
            render_document.RenderDocument(style, md, out=out or log.out, notes=notes, title=title, command_metadata=meta_data, command_node=command_node)
            metrics.Ran()
            if out:
                console_io.More(out.getvalue())
            sys.exit(0)
    return Action