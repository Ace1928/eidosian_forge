from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import config as configuration
from googlecloudsdk.core import config as gcloud_config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import encoding
import six
def _GetKeyBindingsHelp():
    """Returns the function key bindings help markdown."""
    if six.PY2:
        return ''
    lines = []
    for key in bindings.KeyBindings().bindings:
        help_text = key.GetHelp(markdown=True)
        if help_text:
            lines.append('\n{}:::'.format(key.GetLabel(markdown=True)))
            lines.append(help_text)
    return '\n'.join(lines)