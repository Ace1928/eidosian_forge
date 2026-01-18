import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wandb
from wandb.apis.internal import Api
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.launch.builder.build import get_current_python_version
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.utils import _is_git_uri
from wandb.sdk.lib import filesystem
from wandb.util import make_artifact_name_safe
def _dump_metadata_and_requirements(tmp_path: str, metadata: Dict[str, Any], requirements: Optional[List[str]]) -> None:
    """Dump manufactured metadata and requirements.txt.

    File used by the job_builder to create a job from provided metadata.
    """
    filesystem.mkdir_exists_ok(tmp_path)
    with open(os.path.join(tmp_path, 'wandb-metadata.json'), 'w') as f:
        json.dump(metadata, f)
    requirements = requirements or []
    with open(os.path.join(tmp_path, 'requirements.txt'), 'w') as f:
        f.write('\n'.join(requirements))