import threading
from typing import MutableMapping, NamedTuple
import wandb
class FileCountsByCategory(NamedTuple):
    artifact: int
    wandb: int
    media: int
    other: int