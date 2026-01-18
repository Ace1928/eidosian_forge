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
def _has_git_job_ingredients(self, metadata: Dict[str, Any]) -> bool:
    git_info: Dict[str, str] = metadata.get('git', {})
    if self._is_notebook_run and metadata.get('root') is None:
        return False
    return git_info.get('remote') is not None and git_info.get('commit') is not None