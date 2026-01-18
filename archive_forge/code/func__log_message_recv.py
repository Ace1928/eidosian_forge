import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _log_message_recv(msg: 'MessageType', t: 'TransportType') -> None:
    _log_message(msg, 'recv')