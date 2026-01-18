import sys
from typing import Any, Dict, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.integration.keras.keras import patch_tf_keras
from wandb.sdk.lib import telemetry
def _get_lr(self) -> Union[float, None]:
    if isinstance(self.model.optimizer.learning_rate, tf.Variable):
        return float(self.model.optimizer.learning_rate.numpy().item())
    try:
        return float(self.model.optimizer.learning_rate(step=self.global_step).numpy().item())
    except Exception:
        wandb.termerror('Unable to log learning rate.', repeat=False)
        return None