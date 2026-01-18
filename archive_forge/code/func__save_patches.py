import datetime
import glob
import json
import logging
import os
import subprocess
import sys
from shutil import copyfile
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.filenames import CONDA_ENVIRONMENTS_FNAME, DIFF_FNAME, METADATA_FNAME
from wandb.sdk.lib.gitlib import GitRepo
from .assets.interfaces import Interface
def _save_patches(self) -> None:
    """Save the current state of this repository to one or more patches.

        Makes one patch against HEAD and another one against the most recent
        commit that occurs in an upstream branch. This way we can be robust
        to history editing as long as the user never does "push -f" to break
        history on an upstream branch.

        Writes the first patch to <files_dir>/<DIFF_FNAME> and the second to
        <files_dir>/upstream_diff_<commit_id>.patch.

        """
    if not self.git.enabled:
        return None
    logger.debug('Saving git patches')
    try:
        root = self.git.root
        diff_args = ['git', 'diff']
        if self.git.has_submodule_diff:
            diff_args.append('--submodule=diff')
        if self.git.dirty:
            patch_path = os.path.join(self.settings.files_dir, DIFF_FNAME)
            with open(patch_path, 'wb') as patch:
                subprocess.check_call(diff_args + ['HEAD'], stdout=patch, cwd=root, timeout=5)
                self.saved_patches.append(os.path.relpath(patch_path, start=self.settings.files_dir))
        upstream_commit = self.git.get_upstream_fork_point()
        if upstream_commit and upstream_commit != self.git.repo.head.commit:
            sha = upstream_commit.hexsha
            upstream_patch_path = os.path.join(self.settings.files_dir, f'upstream_diff_{sha}.patch')
            with open(upstream_patch_path, 'wb') as upstream_patch:
                subprocess.check_call(diff_args + [sha], stdout=upstream_patch, cwd=root, timeout=5)
                self.saved_patches.append(os.path.relpath(upstream_patch_path, start=self.settings.files_dir))
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error('Error generating diff: %s' % e)
    logger.debug('Saving git patches done')