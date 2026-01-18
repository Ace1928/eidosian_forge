from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def ExtendValues(self, values, perm, selected):
    """Add selected values to a template and extend the selected rows."""
    vals = [row[self.column] for row in selected]
    log.info('cache collection={} adding values={}'.format(self.collection, vals))
    v = [perm + [val] for val in vals]
    values.extend(v)