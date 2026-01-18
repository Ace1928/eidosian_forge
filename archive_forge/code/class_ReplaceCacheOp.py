from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
class ReplaceCacheOp(_UpdateCacheOp):
    """A ReplaceCache operation."""

    def UpdateRows(self, table, rows):
        """Replaces table with rows."""
        table.DeleteRows()
        table.AddRows(rows)