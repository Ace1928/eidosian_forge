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
def GetDiffFiles(self, restrict=None):
    """Print a list of help text files that are distinct from source, if any."""
    with file_utils.TemporaryDirectory() as temp_dir:
        walker = self._generator(self._cli, temp_dir, None, restrict=restrict)
        walker.Walk(hidden=True)
        diff = HelpAccumulator(restrict=restrict)
        DirDiff(self._directory, temp_dir, diff)
        return sorted(diff.GetChanges())