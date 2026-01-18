from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.core.updater import installers
def _get_absolute_position(self):
    """Returns absolute position in the stream.

    Hashing and progress reporting logic relies on absolute positions. Since
    child classes overwrite `tell` to make it return relative positions, we need
    to write hashing and progress reporting in a way that does not reference
    `self.tell`, which this function makes possible.
    """
    return self._stream.tell()