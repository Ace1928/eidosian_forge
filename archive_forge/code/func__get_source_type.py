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
def _get_source_type(self, metadata: Dict[str, Any]) -> Optional[str]:
    if self._source_type:
        return self._source_type
    if self._has_git_job_ingredients(metadata):
        _logger.info('is repo sourced job')
        return 'repo'
    if self._has_artifact_job_ingredients():
        _logger.info('is artifact sourced job')
        return 'artifact'
    if self._has_image_job_ingredients(metadata):
        _logger.info('is image sourced job')
        return 'image'
    _logger.info('no source found')
    return None