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
def _ImplementationCreateTable(self, name, columns, keys):
    """sqlite3 implementation specific _CreateTable."""
    field_list = [_FieldRef(i) for i in range(columns)]
    key_list = [_FieldRef(i) for i in range(keys or 1)]
    field_list.append('PRIMARY KEY ({keys})'.format(keys=', '.join(key_list)))
    fields = '({fields})'.format(fields=', '.join(field_list))
    self.cursor.execute('CREATE TABLE IF NOT EXISTS "{name}" {fields}'.format(name=name, fields=fields))