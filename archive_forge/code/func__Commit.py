from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import gc
import os
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import metadata_table
from googlecloudsdk.core.cache import persistent_cache_base
from googlecloudsdk.core.util import files
import six
from six.moves import range  # pylint: disable=redefined-builtin
import sqlite3
def _Commit(self):
    """Commits changed/deleted table data."""
    if self.changed:
        self.changed = False
        if self.deleted:
            self.deleted = False
            self._cache._metadata.DeleteRows([(self.name,)])
            del self._cache._tables[self.name]
        else:
            self._cache._metadata.AddRows([metadata_table.Metadata.Row(name=self.name, columns=self.columns, keys=self.keys, timeout=self.timeout, modified=self.modified, restricted=self.restricted, version=self._cache.version)])