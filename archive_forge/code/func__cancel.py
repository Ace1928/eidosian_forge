import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _cancel(self) -> None:
    mailbox_slot = self.address
    if self._interface:
        self._interface.publish_cancel(mailbox_slot)