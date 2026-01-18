import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def _authenticate_plain(self, domain: str, username: str, password: str) -> Tuple[bool, bytes]:
    """PLAIN ZAP authentication"""
    allowed = False
    reason = b''
    if self.passwords:
        if not domain:
            domain = '*'
        if domain in self.passwords:
            if username in self.passwords[domain]:
                if password == self.passwords[domain][username]:
                    allowed = True
                else:
                    reason = b'Invalid password'
            else:
                reason = b'Invalid username'
        else:
            reason = b'Invalid domain'
        if allowed:
            self.log.debug('ALLOWED (PLAIN) domain=%s username=%s password=%s', domain, username, password)
        else:
            self.log.debug('DENIED %s', reason)
    else:
        reason = b'No passwords defined'
        self.log.debug('DENIED (PLAIN) %s', reason)
    return (allowed, reason)