import logging
from typing import TYPE_CHECKING, Any, Optional
from ..lib.mailbox import Mailbox
from ..lib.sock_client import SockClient
from .interface_shared import InterfaceShared
from .message_future import MessageFuture
from .router_sock import MessageSockRouter
def _publish(self, record: 'pb.Record', local: Optional[bool]=None) -> None:
    self._assign(record)
    self._sock_client.send_record_publish(record)