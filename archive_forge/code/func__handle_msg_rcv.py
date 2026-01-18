import logging
import threading
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional
from ..lib import mailbox, tracelog
from .message_future import MessageFuture
def _handle_msg_rcv(self, msg: 'pb.Result') -> None:
    if self._mailbox and msg.control.mailbox_slot:
        self._mailbox.deliver(msg)
        return
    with self._lock:
        future = self._pending_reqs.pop(msg.uuid, None)
    if future is None:
        if msg.uuid != '':
            tracelog.log_message_assert(msg)
            logger.warning('No listener found for msg with uuid %s (%s)', msg.uuid, msg)
        return
    future._set_object(msg)