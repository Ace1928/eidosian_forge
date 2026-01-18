from typing import Optional, Sequence, Union
import wandb
from wandb.errors import UnsupportedError
from wandb.sdk import wandb_run
from wandb.sdk.lib.wburls import wburls
def _require_service(self) -> None:
    wandb.teardown = wandb._teardown
    wandb.attach = wandb._attach
    wandb_run.Run.detach = wandb_run.Run._detach