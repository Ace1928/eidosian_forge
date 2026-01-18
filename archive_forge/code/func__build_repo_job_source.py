import json
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.lib.filenames import DIFF_FNAME, METADATA_FNAME, REQUIREMENTS_FNAME
from wandb.util import make_artifact_name_safe
from .settings_static import SettingsStatic
def _build_repo_job_source(self, program_relpath: str, metadata: Dict[str, Any]) -> Tuple[Optional[GitSourceDict], Optional[str]]:
    git_info: Dict[str, str] = metadata.get('git', {})
    remote = git_info.get('remote')
    commit = git_info.get('commit')
    root = metadata.get('root')
    assert remote is not None
    assert commit is not None
    if self._is_notebook_run:
        if not os.path.exists(os.path.join(os.getcwd(), os.path.basename(program_relpath))):
            return (None, None)
        if root is None or self._settings._jupyter_root is None:
            _logger.info('target path does not exist, exiting')
            return (None, None)
        assert self._settings._jupyter_root is not None
        full_program_path = os.path.join(os.path.relpath(str(self._settings._jupyter_root), root), program_relpath)
        full_program_path = os.path.normpath(full_program_path)
        if full_program_path.startswith('..'):
            split_path = full_program_path.split('/')
            count_dots = 0
            for p in split_path:
                if p == '..':
                    count_dots += 1
            full_program_path = '/'.join(split_path[2 * count_dots:])
    else:
        full_program_path = program_relpath
    entrypoint = self._get_entrypoint(full_program_path, metadata)
    source: GitSourceDict = {'git': {'remote': remote, 'commit': commit}, 'entrypoint': entrypoint, 'notebook': self._is_notebook_run}
    name = self._make_job_name(f'{remote}_{program_relpath}')
    return (source, name)