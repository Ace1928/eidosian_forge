import logging
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_internal_pb2 as pb
from ..lib.mailbox import Mailbox
from .interface_queue import InterfaceQueue
from .router_relay import MessageRelayRouter
def _init_router(self) -> None:
    if self.record_q and self.result_q and self.relay_q:
        self._router = MessageRelayRouter(request_queue=self.record_q, response_queue=self.result_q, relay_queue=self.relay_q, mailbox=self._mailbox)