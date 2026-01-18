from typing import TYPE_CHECKING, Optional
from ..lib.mailbox import Mailbox
from ..lib.sock_client import SockClient, SockClientClosedError
from .router import MessageRouter, MessageRouterClosedError
def _read_message(self) -> Optional['pb.Result']:
    try:
        resp = self._sock_client.read_server_response(timeout=1)
    except SockClientClosedError:
        raise MessageRouterClosedError
    if not resp:
        return None
    msg = resp.result_communicate
    return msg