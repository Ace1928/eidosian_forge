from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import persistent_cache_base
import six
def InitializeMetadata(self):
    """Initializes the metadata table and self._metadata."""
    self.Table('__metadata__', restricted=True, columns=Metadata.COLUMNS, keys=1, timeout=0)