from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import fnmatch
import json
import os
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import metadata_table
from googlecloudsdk.core.cache import persistent_cache_base
from googlecloudsdk.core.util import files
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _RowMatch(self, row_template, row):
    """Returns True if row_template matches row."""
    if row_template:
        for i in range(len(row_template)):
            if row_template[i] is not None:
                if isinstance(row_template[i], six.string_types) and isinstance(row[i], six.string_types):
                    if not fnmatch.fnmatch(row[i], row_template[i]):
                        return False
                elif row_template[i] != row[i]:
                    return False
    return True