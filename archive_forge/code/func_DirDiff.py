from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
import shutil
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import text
import six
def DirDiff(old_dir, new_dir, diff):
    """Calls diff.AddChange(op, file) on files that changed from old_dir new_dir.

  diff.AddChange() can construct the {'add', 'delete', 'edit'} file operations
  that convert old_dir to match new_dir. Directory differences are ignored.

  Args:
    old_dir: The old directory path name.
    new_dir: The new directory path name.
    diff: A DiffAccumulator instance.

  Returns:
    The return value of the first diff.AddChange() call that returns non-zero
    or None if all diff.AddChange() calls returned zero.
  """
    with TimeIt('GetDirFilesRecursive new files'):
        new_files = GetDirFilesRecursive(new_dir)
    with TimeIt('GetDirFilesRecursive old files'):
        old_files = GetDirFilesRecursive(old_dir)

    def _FileDiff(file):
        """Diffs a file in new_dir and old_dir."""
        new_contents, new_binary = GetFileContents(os.path.join(new_dir, file))
        if not new_binary:
            diff.Validate(file, new_contents)
        if file in old_files:
            old_contents, old_binary = GetFileContents(os.path.join(old_dir, file))
            if old_binary == new_binary and old_contents == new_contents:
                return
            return ('edit', file, old_contents, new_contents)
        else:
            return ('add', file, None, new_contents)
    with parallel.GetPool(16) as pool:
        results = []
        for file in new_files:
            if diff.Ignore(file):
                continue
            result = pool.ApplyAsync(_FileDiff, (file,))
            results.append(result)
        for result_future in results:
            result = result_future.Get()
            if result:
                op, file, old_contents, new_contents = result
                prune = diff.AddChange(op, file, old_contents, new_contents)
                if prune:
                    return prune
    for file in old_files:
        if diff.Ignore(file):
            continue
        if file not in new_files:
            prune = diff.AddChange('delete', file)
            if prune:
                return prune
    return None