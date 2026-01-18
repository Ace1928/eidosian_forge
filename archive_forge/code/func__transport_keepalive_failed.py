import logging
import time
from abc import abstractmethod
from multiprocessing.process import BaseProcess
from typing import Any, Optional, cast
import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.util import json_dumps_safer, json_friendly
from ..lib.mailbox import Mailbox, MailboxHandle
from .interface import InterfaceBase
from .message_future import MessageFuture
from .router import MessageRouter
def _transport_keepalive_failed(self, keepalive_interval: int=5) -> bool:
    if self._transport_failed:
        return True
    now = time.monotonic()
    if now < self._transport_success_timestamp + keepalive_interval:
        return False
    try:
        self.publish_keepalive()
    except Exception:
        self._transport_mark_failed()
    else:
        self._transport_mark_success()
    return self._transport_failed