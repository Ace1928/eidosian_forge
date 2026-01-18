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
def _publish_defer(self, state: 'pb.DeferRequest.DeferState.V') -> None:
    defer = pb.DeferRequest(state=state)
    rec = self._make_request(defer=defer)
    self._publish(rec, local=True)