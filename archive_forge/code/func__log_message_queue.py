import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _log_message_queue(msg: 'MessageQueueType', q: 'QueueType') -> None:
    _annotate_message(msg)
    resource = getattr(q, ANNOTATE_QUEUE_NAME, None)
    _log_message(msg, 'queue', resource=resource)