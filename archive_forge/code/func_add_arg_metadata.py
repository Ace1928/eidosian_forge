from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def add_arg_metadata(self, arg_metadata):
    """Adds an argument's metadata to comprehensive metadata list.

    Args:
      arg_metadata: The argument metadata to be added.
    """
    self._argument_metadatas.append(arg_metadata)