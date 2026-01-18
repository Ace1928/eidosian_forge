import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def _authenticate_gssapi(self, domain: str, principal: bytes) -> Tuple[bool, bytes]:
    """Nothing to do for GSSAPI, which has already been handled by an external service."""
    self.log.debug('ALLOWED (GSSAPI) domain=%s principal=%s', domain, principal)
    return (True, b'OK')