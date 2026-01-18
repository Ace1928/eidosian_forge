from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _WrapDescription(depth, text):
    """Returns description: |- text wrapped so it won't exceed _WIDTH at depth.

  The YAML representer doesn't seem to take the length of the current tag
  into account when deciding whether to inline strings or use |-. In this case
  the tag is always "description: ". This function detects when YAML would fail
  and adds temporary marker lines to produce the desired output. The marker
  lines are removed prior to final output.

  Args:
    depth: The nested dict depth.
    text: The text string to wrap.

  Returns:
    Text wrapped so it won't exceed _WIDTH at depth.
  """
    width = _WIDTH - depth * _INDENT
    lines = textwrap.wrap(text, max(_MINWRAP, width))
    if len(lines) != 1:
        return '\n'.join(lines)
    line = lines[0]
    nudge = width - (len(line) + _DESCRIPTION_INDENT)
    if nudge < 0:
        return line + '\n' + nudge * ' ' + _YAML_WORKAROUND
    return line