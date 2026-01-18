from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _LegacyPostProcess(self, snapshot=None):
    """Runs the gcloud command to post process the update.

    This runs gcloud as a subprocess so that the new version of gcloud (the one
    we just updated to) is run instead of the old code (which is running here).
    We do this so the new code can say how to correctly post process itself.

    Args:
      snapshot: ComponentSnapshot, The component snapshot for the version
        we are updating do. The location of gcloud and the command to run can
        change from version to version, which is why we try to pull this
        information from the latest snapshot.  For a restore operation, we don't
        have that information so we fall back to a best effort default.
    """
    log.debug('Legacy post-processing...')
    command = None
    gcloud_path = None
    if snapshot:
        if snapshot.sdk_definition.post_processing_command:
            command = snapshot.sdk_definition.post_processing_command.split(' ')
        if snapshot.sdk_definition.gcloud_rel_path:
            gcloud_path = os.path.join(self.__sdk_root, snapshot.sdk_definition.gcloud_rel_path)
    command = command or ['components', 'post-process']
    if self.__skip_compile_python:
        command.append('--no-compile-python')
    gcloud_path = gcloud_path or config.GcloudPath()
    args = execution_utils.ArgsForPythonTool(gcloud_path, *command)
    try:
        with progress_tracker.ProgressTracker(message='Performing post processing steps', tick_delay=0.25):
            try:
                ret_val = execution_utils.Exec(args, no_exit=True, out_func=log.file_only_logger.debug, err_func=log.file_only_logger.debug)
            except (OSError, execution_utils.InvalidCommandError, execution_utils.PermissionError):
                log.debug('Failed to execute post-processing command', exc_info=True)
                raise PostProcessingError()
            if ret_val:
                log.debug('Post-processing command exited non-zero')
                raise PostProcessingError()
    except PostProcessingError:
        log.warning('Post processing failed.  Run `gcloud info --show-log` to view the failures.')