import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _log_message_link(src: 'MessageType', dest: 'MessageType') -> None:
    _log_message(src, 'source')
    _log_message(dest, 'dest')