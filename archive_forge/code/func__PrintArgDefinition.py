from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _PrintArgDefinition(self, arg, depth=0, single=False):
    """Prints a positional or flag arg definition list item at depth."""
    usage = usage_text.GetArgUsage(arg, definition=True, markdown=True)
    if not usage:
        return
    self._out('\n{usage}{depth}\n'.format(usage=usage, depth=':' * (depth + _SECOND_LINE_OFFSET)))
    if arg.is_required and depth and (not single):
        modal = '\n+\nThis {arg_type} argument must be specified if any of the other arguments in this group are specified.'.format(arg_type=self._ArgTypeName(arg))
    else:
        modal = ''
    details = self.GetArgDetails(arg, depth=depth).replace('\n\n', '\n+\n')
    self._out('\n{details}{modal}\n'.format(details=details, modal=modal))