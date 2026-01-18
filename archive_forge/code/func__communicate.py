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
def _communicate(self, rec: pb.Record, timeout: Optional[int]=5, local: Optional[bool]=None) -> Optional[pb.Result]:
    return self._communicate_async(rec, local=local).get(timeout=timeout)