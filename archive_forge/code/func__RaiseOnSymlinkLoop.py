from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.util import glob
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _RaiseOnSymlinkLoop(self, full_path):
    """Raise SymlinkLoopError if the given path is a symlink loop."""
    if not os.path.islink(encoding.Encode(full_path, encoding='utf-8')):
        return
    p = os.readlink(full_path)
    targets = set()
    while os.path.islink(p):
        if p in targets:
            raise SymlinkLoopError('The symlink [{}] refers to itself.'.format(full_path))
        targets.add(p)
        p = os.readlink(p)
    p = os.path.dirname(full_path)
    while p and os.path.basename(p):
        if os.path.samefile(p, full_path):
            raise SymlinkLoopError('The symlink [{}] refers to its own containing directory.'.format(full_path))
        p = os.path.dirname(p)