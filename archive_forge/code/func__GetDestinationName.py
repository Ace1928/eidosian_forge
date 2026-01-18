from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _GetDestinationName(self, item, dest):
    """Computes the destination name to copy item to.."""
    expander = self._GetExpander(dest)
    if dest.is_dir_like:
        item_dest = dest.Join(os.path.basename(item.path.rstrip('/').rstrip('\\')))
        if item.is_dir_like:
            item_dest = item_dest.Join('')
        if expander.IsFile(dest.path):
            raise DestinationDirectoryExistsError('Cannot copy [{}] to [{}]: [{}] exists and is a file.'.format(item.path, item_dest.path, dest.path))
    else:
        item_dest = dest
    check_func = expander.Exists if item.is_dir_like else expander.IsDir
    if check_func(item_dest.path):
        raise DestinationDirectoryExistsError('Cannot copy [{}] to [{}]: The destination already exists. If you meant to copy under this destination, add a slash to the end of its path.'.format(item.path, item_dest.path))
    return item_dest