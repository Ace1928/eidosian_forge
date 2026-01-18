from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetPrintFormat(self, margin=0, width=0):
    """Constructs and returns a resource_printer print format.

    Args:
      margin: Right hand side padding when one or more columns are wrapped.
      width: The table width.

    Returns:
      The resource printer format string.
    """
    fmt = ['table']
    attr = []
    if self.box:
        attr.append('box')
    if not self.heading:
        attr.append('no-heading')
    if margin:
        attr.append('margin={}'.format(margin))
    if width:
        attr.append('width={}'.format(width))
    if attr:
        fmt.append('[' + ','.join(attr) + ']')
    fmt.append('(')
    for index, column in enumerate(self.columns):
        if index:
            fmt.append(',')
        fmt.append('[{}]:label={}:align={}'.format(index, repr(column.label or '').lstrip('u'), column.align))
        if column.width:
            fmt.append(':width={}'.format(column.width))
    if margin:
        fmt.append(':wrap')
    fmt.append(')')
    return ''.join(fmt)