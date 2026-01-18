import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _set_wait_all(self, wait_all: _MailboxWaitAll) -> None:
    assert not self._wait_all, 'Only one caller can wait_all for a slot at a time'
    self._wait_all = wait_all