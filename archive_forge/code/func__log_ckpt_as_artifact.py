import os
import string
import sys
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath
from ..keras import patch_tf_keras
def _log_ckpt_as_artifact(self, filepath: str, aliases: Optional[List[str]]=None) -> None:
    """Log model checkpoint as  W&B Artifact."""
    try:
        assert wandb.run is not None
        model_checkpoint_artifact = wandb.Artifact(f'run_{wandb.run.id}_model', type='model')
        if os.path.isfile(filepath):
            model_checkpoint_artifact.add_file(filepath)
        elif os.path.isdir(filepath):
            model_checkpoint_artifact.add_dir(filepath)
        else:
            raise FileNotFoundError(f'No such file or directory {filepath}')
        wandb.log_artifact(model_checkpoint_artifact, aliases=aliases or [])
    except ValueError:
        pass