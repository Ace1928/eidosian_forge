from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
@six.add_metaclass(abc.ABCMeta)
class _UpdateCacheOp(object):
    """The cache update operation base class."""

    def __init__(self, completer):
        self._completer_class = completer

    def Update(self, uris):
        """Applies UpdateRows() to tables that contain the resources uris."""
        try:
            with resource_cache.ResourceCache() as cache:
                completer = self._completer_class(cache=cache)
                tables = {}
                for uri in uris:
                    row = completer.StringToRow(uri)
                    table = completer.GetTableForRow(row)
                    entry = tables.get(table.name)
                    if not entry:
                        entry = _TableRows(table)
                        tables[table.name] = entry
                    entry.rows.append(row)
                for table, rows in six.iteritems(tables):
                    self.UpdateRows(table, rows)
        except Exception:
            pass

    @abc.abstractmethod
    def UpdateRows(self, table, rows):
        """Updates table with rows."""
        pass