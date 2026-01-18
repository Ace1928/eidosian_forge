import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
@property
def _is_stopped(self) -> bool:
    return self._stopped