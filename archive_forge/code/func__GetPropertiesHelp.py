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
def _GetPropertiesHelp():
    """Returns the properties help markdown."""
    lines = []
    for prop in sorted(properties.VALUES.interactive, key=lambda p: p.name):
        if prop.help_text:
            lines.append('\n*{}*::'.format(prop.name))
            lines.append(prop.help_text)
            default = prop.default
            if default is not None:
                if isinstance(default, six.string_types):
                    default = '"{}"'.format(default)
                else:
                    if default in (False, True):
                        default = six.text_type(default).lower()
                    default = '*{}*'.format(default)
                lines.append('The default value is {}.'.format(default))
    return '\n'.join(lines)