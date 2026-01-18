from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import sys
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class _RcUpdater(object):
    """Updates the RC file completion and PATH code injection."""

    def __init__(self, completion_update, path_update, shell, rc_path, sdk_root):
        self.completion_update = completion_update
        self.path_update = path_update
        self.rc_path = rc_path
        compatible_shell = _COMPATIBLE_INC_SHELL.get(shell, shell)
        self.completion = os.path.join(sdk_root, 'completion.{shell}.inc'.format(shell=compatible_shell))
        self.path = os.path.join(sdk_root, 'path.{shell}.inc'.format(shell=compatible_shell))
        self.shell = shell

    def _CompletionExists(self):
        return os.path.exists(self.completion)

    def Update(self):
        """Creates or updates the RC file."""
        if self.rc_path:
            if os.path.isfile(self.rc_path):
                rc_contents = files.ReadFileContents(self.rc_path)
                original_rc_contents = rc_contents
            elif os.path.exists(self.rc_path):
                _TraceAction('[{rc_path}] exists and is not a file, so it cannot be updated.'.format(rc_path=self.rc_path))
                return
            else:
                rc_contents = ''
                original_rc_contents = ''
            if self.path_update:
                rc_contents = _GetRcContents('# The next line updates PATH for the Google Cloud SDK.', self.path, rc_contents, self.shell)
            if self.completion_update and self._CompletionExists():
                rc_contents = _GetRcContents('# The next line enables shell command completion for gcloud.', self.completion, rc_contents, self.shell, pattern='# The next line enables [a-z][a-z]* command completion for gcloud.')
            if rc_contents == original_rc_contents:
                _TraceAction('No changes necessary for [{rc}].'.format(rc=self.rc_path))
                return
            if os.path.exists(self.rc_path):
                rc_backup = self.rc_path + '.backup'
                _TraceAction('Backing up [{rc}] to [{backup}].'.format(rc=self.rc_path, backup=rc_backup))
                shutil.copyfile(self.rc_path, rc_backup)
            rc_dir = os.path.dirname(self.rc_path)
            try:
                files.MakeDir(rc_dir)
            except (files.Error, IOError, OSError):
                _TraceAction('Could not create directories for [{rc_path}], so it cannot be updated.'.format(rc_path=self.rc_path))
                return
            try:
                files.WriteFileContents(self.rc_path, rc_contents)
            except (files.Error, IOError, OSError):
                _TraceAction('Could not update [{rc_path}]. Ensure you have write access to this location.'.format(rc_path=self.rc_path))
                return
            _TraceAction('[{rc_path}] has been updated.'.format(rc_path=self.rc_path))
            _TraceAction(console_io.FormatRequiredUserAction('Start a new shell for the changes to take effect.'))
        screen_reader = properties.VALUES.accessibility.screen_reader.GetBool()
        prefix = '' if screen_reader else '==> '
        if not self.completion_update and self._CompletionExists():
            _TraceAction(prefix + 'Source [{rc}] in your profile to enable shell command completion for gcloud.'.format(rc=self.completion))
        if not self.path_update:
            _TraceAction(prefix + 'Source [{rc}] in your profile to add the Google Cloud SDK command line tools to your $PATH.'.format(rc=self.path))