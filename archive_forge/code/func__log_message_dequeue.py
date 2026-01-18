import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _log_message_dequeue(msg: 'MessageQueueType', q: 'QueueType') -> None:
    resource = getattr(q, ANNOTATE_QUEUE_NAME, None)
    _log_message(msg, 'dequeue', resource=resource)