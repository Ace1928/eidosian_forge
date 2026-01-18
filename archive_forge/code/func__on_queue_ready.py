from __future__ import annotations
import json
from botocore.serialize import Serializer
from vine import transform
from kombu.asynchronous.aws.connection import AsyncAWSQueryConnection
from kombu.asynchronous.aws.ext import AWSRequest
from .ext import boto3
from .message import AsyncMessage
from .queue import AsyncQueue
def _on_queue_ready(self, name, queues):
    return next((q for q in queues if q.url.endswith(name)), None)