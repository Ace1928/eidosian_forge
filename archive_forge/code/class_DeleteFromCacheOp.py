from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
class DeleteFromCacheOp(_UpdateCacheOp):
    """A DeleteFromCache operation."""

    def UpdateRows(self, table, rows):
        """Deletes rows from table."""
        table.DeleteRows(rows)