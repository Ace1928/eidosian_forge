from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _Visible(self, row):
    """Return the visible list items in row."""
    if not self._visible or not row:
        return row
    visible = []
    for index in self._visible:
        visible.append(row[index])
    return visible