from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_property
def _SetVisibleColumns(self):
    """Sets visible columns list if column attributes have been loaded."""
    if self.column_attributes:
        self._is_column_visible = [not column.attribute.hidden for column in self.column_attributes.Columns()]
    else:
        self._is_column_visible = None