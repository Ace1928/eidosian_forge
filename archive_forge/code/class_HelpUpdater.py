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
class HelpUpdater(object):
    """Updates the document directory to match the current CLI.

  Attributes:
    _cli: The Current CLI.
    _directory: The help document directory.
    _generator: The document generator.
    _hidden: Boolean indicating whether to update hidden commands.
    _test: Show but do not apply operations if True.
  """

    def __init__(self, cli, directory, generator, test=False, hidden=False):
        """Constructor.

    Args:
      cli: The Current CLI.
      directory: The help document directory.
      generator: An uninstantiated walker_util document generator.
      test: Show but do not apply operations if True.
      hidden: Boolean indicating whether the hidden commands should be used.

    Raises:
      HelpUpdateError: If the destination directory does not exist.
    """
        if not os.path.isabs(directory):
            raise HelpUpdateError('Destination directory [%s] must be absolute.' % directory)
        self._cli = cli
        self._directory = directory
        self._generator = generator
        self._hidden = hidden
        self._test = test

    def _Update(self, restrict):
        """Update() helper method. Returns the number of changed help doc files."""
        with file_utils.TemporaryDirectory() as temp_dir:
            pb = console_io.ProgressBar(label='Generating Help Document Files')
            with TimeIt('Creating walker'):
                walker = self._generator(self._cli, temp_dir, pb.SetProgress, restrict=restrict)
            start = time.time()
            pb.Start()
            walker.Walk(hidden=True)
            pb.Finish()
            elapsed_time = time.time() - start
            log.info('Generating Help Document Files took {}'.format(elapsed_time))
            diff = HelpAccumulator(restrict=restrict)
            with TimeIt('Diffing'):
                DirDiff(self._directory, temp_dir, diff)
            ops = collections.defaultdict(list)
            changes = 0
            with TimeIt('Getting diffs'):
                for op, path in sorted(diff.GetChanges()):
                    changes += 1
                    if not self._test or changes < TEST_CHANGES_DISPLAY_MAX:
                        log.status.Print('{0} {1}'.format(op, path))
                    ops[op].append(path)
            if self._test:
                if changes:
                    if changes >= TEST_CHANGES_DISPLAY_MAX:
                        log.status.Print('...')
                    log.status.Print('{0} help text {1} changed'.format(changes, text.Pluralize(changes, 'file')))
                return changes
            with TimeIt('Updating destination files'):
                for op in ('add', 'edit', 'delete'):
                    for path in ops[op]:
                        dest_path = os.path.join(self._directory, path)
                        if op in ('add', 'edit'):
                            if op == 'add':
                                subdir = os.path.dirname(dest_path)
                                if subdir:
                                    file_utils.MakeDir(subdir)
                            temp_path = os.path.join(temp_dir, path)
                            shutil.copyfile(temp_path, dest_path)
                        elif op == 'delete':
                            try:
                                os.remove(dest_path)
                            except OSError:
                                pass
            return changes

    def Update(self, restrict=None):
        """Updates the help document directory to match the current CLI.

    Args:
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups.

    Raises:
      HelpUpdateError: If the destination directory does not exist.

    Returns:
      The number of changed help document files.
    """
        if not os.path.isdir(self._directory):
            raise HelpUpdateError('Destination directory [%s] must exist and be searchable.' % self._directory)
        try:
            return self._Update(restrict)
        except (IOError, OSError, SystemError) as e:
            raise HelpUpdateError('Update failed: %s' % six.text_type(e))

    def GetDiffFiles(self, restrict=None):
        """Print a list of help text files that are distinct from source, if any."""
        with file_utils.TemporaryDirectory() as temp_dir:
            walker = self._generator(self._cli, temp_dir, None, restrict=restrict)
            walker.Walk(hidden=True)
            diff = HelpAccumulator(restrict=restrict)
            DirDiff(self._directory, temp_dir, diff)
            return sorted(diff.GetChanges())