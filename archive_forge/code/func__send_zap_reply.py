import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def _send_zap_reply(self, request_id: bytes, status_code: bytes, status_text: bytes, user_id: str='anonymous') -> None:
    """Send a ZAP reply to finish the authentication."""
    user_id = user_id if status_code == b'200' else b''
    if isinstance(user_id, str):
        user_id = user_id.encode(self.encoding, 'replace')
    metadata = b''
    self.log.debug('ZAP reply code=%s text=%s', status_code, status_text)
    reply = [VERSION, request_id, status_code, status_text, user_id, metadata]
    self.zap_socket.send_multipart(reply)