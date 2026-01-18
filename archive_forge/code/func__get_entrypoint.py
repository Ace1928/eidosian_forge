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
def _get_entrypoint(self, program_relpath: str, metadata: Dict[str, Any]) -> List[str]:
    if metadata.get('_partial'):
        if metadata.get('entrypoint'):
            entrypoint: List[str] = metadata['entrypoint']
            return entrypoint
        assert metadata.get('python')
        python = metadata['python']
        if python.count('.') > 1:
            python = '.'.join(python.split('.')[:2])
        entrypoint = [f'python{python}', program_relpath]
        return entrypoint
    entrypoint = [os.path.basename(sys.executable), program_relpath]
    return entrypoint