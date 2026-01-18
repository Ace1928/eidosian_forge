import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _mark_handle_failed(self, handle: 'MailboxHandle') -> None:
    handle._mark_failed()
    self._failed_handles += 1