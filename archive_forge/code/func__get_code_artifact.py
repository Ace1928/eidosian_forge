import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.errors import CommError
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.data_types._dtypes import InvalidType, Type, TypeRegistry
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
def _get_code_artifact(self, artifact_string):
    artifact_string, base_url, is_id = util.parse_artifact_string(artifact_string)
    if is_id:
        code_artifact = wandb.Artifact._from_id(artifact_string, self._api._client)
    else:
        code_artifact = self._api.artifact(name=artifact_string, type='code')
    if code_artifact is None:
        raise LaunchError('No code artifact found')
    if code_artifact.state == ArtifactState.DELETED:
        raise LaunchError(f'Job {self.name} references deleted code artifact {code_artifact.name}')
    return code_artifact