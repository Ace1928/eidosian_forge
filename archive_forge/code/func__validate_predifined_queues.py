from SQS when you have short-running tasks (or a large number of workers).
from __future__ import annotations
import base64
import socket
import string
import uuid
from datetime import datetime
from queue import Empty
from botocore.client import Config
from botocore.exceptions import ClientError
from vine import ensure_promise, promise, transform
from kombu.asynchronous import get_event_loop
from kombu.asynchronous.aws.ext import boto3, exceptions
from kombu.asynchronous.aws.sqs.connection import AsyncSQSConnection
from kombu.asynchronous.aws.sqs.message import AsyncMessage
from kombu.log import get_logger
from kombu.utils import scheduling
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _validate_predifined_queues(self):
    """Check that standard and FIFO queues are named properly.

        AWS requires FIFO queues to have a name
        that ends with the .fifo suffix.
        """
    for queue_name, q in self.predefined_queues.items():
        fifo_url = q['url'].endswith('.fifo')
        fifo_name = queue_name.endswith('.fifo')
        if fifo_url and (not fifo_name):
            raise InvalidQueueException("Queue with url '{}' must have a name ending with .fifo".format(q['url']))
        elif not fifo_url and fifo_name:
            raise InvalidQueueException("Queue with name '{}' is not a FIFO queue: '{}'".format(queue_name, q['url']))