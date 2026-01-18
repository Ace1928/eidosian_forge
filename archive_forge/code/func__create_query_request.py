from __future__ import annotations
import json
from botocore.serialize import Serializer
from vine import transform
from kombu.asynchronous.aws.connection import AsyncAWSQueryConnection
from kombu.asynchronous.aws.ext import AWSRequest
from .ext import boto3
from .message import AsyncMessage
from .queue import AsyncQueue
def _create_query_request(self, operation, params, queue_url, method):
    params = params.copy()
    if operation:
        params['Action'] = operation
    param_payload = {'data': params}
    if method.lower() == 'get':
        param_payload = {'params': params}
    return AWSRequest(method=method, url=queue_url, **param_payload)