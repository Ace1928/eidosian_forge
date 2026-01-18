from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _GenerateLineValue(value, indent_length=0):
    """Returns a formatted, indented line containing the given value."""
    line_format = _LINE_FORMAT % indent_length
    return line_format.format(indent=_INDENT_STRING, line=value)