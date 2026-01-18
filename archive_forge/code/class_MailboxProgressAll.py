import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
class MailboxProgressAll:
    _progress_handles: List[MailboxProgress]

    def __init__(self) -> None:
        self._progress_handles = []

    def add_progress_handle(self, progress_handle: MailboxProgress) -> None:
        self._progress_handles.append(progress_handle)

    def get_progress_handles(self) -> List[MailboxProgress]:
        return [ph for ph in self._progress_handles if not ph._handle._is_failed]