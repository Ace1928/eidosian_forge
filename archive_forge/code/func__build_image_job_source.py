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
def _build_image_job_source(self, metadata: Dict[str, Any]) -> Tuple[ImageSourceDict, str]:
    image_name = metadata.get('docker')
    assert isinstance(image_name, str)
    raw_image_name = image_name
    if ':' in image_name:
        tag = image_name.split(':')[-1]
        if re.fullmatch('([a-zA-Z0-9_\\-\\.]+)', tag):
            raw_image_name = raw_image_name.replace(f':{tag}', '')
            self._aliases += [tag]
    source: ImageSourceDict = {'image': image_name}
    name = self._make_job_name(raw_image_name)
    return (source, name)