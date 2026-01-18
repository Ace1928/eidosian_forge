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
class JobArtifact(Artifact):

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        super().__init__(name, 'placeholder', *args, **kwargs)
        self._type = JOB_ARTIFACT_TYPE