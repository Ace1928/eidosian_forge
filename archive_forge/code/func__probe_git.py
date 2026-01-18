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
def _probe_git(self, data: Dict[str, Any]) -> Dict[str, Any]:
    if self.settings.disable_git:
        return data
    if not self.git.enabled and self.git.auto:
        return data
    logger.debug('Probing git')
    data['git'] = {'remote': self.git.remote_url, 'commit': self.git.last_commit}
    data['email'] = self.git.email
    data['root'] = self.git.root or data.get('root') or os.getcwd()
    logger.debug('Probing git done')
    return data