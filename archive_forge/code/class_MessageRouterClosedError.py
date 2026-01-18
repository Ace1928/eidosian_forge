import logging
import threading
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional
from ..lib import mailbox, tracelog
from .message_future import MessageFuture
class MessageRouterClosedError(Exception):
    """Router has been closed."""
    pass