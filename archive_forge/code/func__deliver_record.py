import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _deliver_record(self, record: pb.Record, interface: 'InterfaceShared') -> MailboxHandle:
    handle = self.get_handle()
    handle._interface = interface
    handle._keepalive = self._keepalive
    record.control.mailbox_slot = handle.address
    try:
        interface._publish(record)
    except Exception:
        interface._transport_mark_failed()
        raise
    interface._transport_mark_success()
    return handle