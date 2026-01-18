import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _annotate_queue(q: 'QueueType', name: str) -> None:
    setattr(q, ANNOTATE_QUEUE_NAME, name)