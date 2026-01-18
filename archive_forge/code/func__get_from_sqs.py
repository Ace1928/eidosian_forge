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
def _get_from_sqs(self, queue_name, queue_url, connection, count=1, callback=None):
    """Retrieve and handle messages from SQS.

        Uses long polling and returns :class:`~vine.promises.promise`.
        """
    return connection.receive_message(queue_name, queue_url, number_messages=count, wait_time_seconds=self.wait_time_seconds, callback=callback)