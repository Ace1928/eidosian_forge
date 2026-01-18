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
def _make_job_name(self, input_str: str) -> str:
    """Use job name from settings if provided, else use programatic name."""
    if self._settings.job_name:
        return self._settings.job_name
    return make_artifact_name_safe(f'job-{input_str}')