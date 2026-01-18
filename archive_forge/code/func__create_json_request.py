from __future__ import annotations
import json
from botocore.serialize import Serializer
from vine import transform
from kombu.asynchronous.aws.connection import AsyncAWSQueryConnection
from kombu.asynchronous.aws.ext import AWSRequest
from .ext import boto3
from .message import AsyncMessage
from .queue import AsyncQueue
def _create_json_request(self, operation, params, queue_url):
    params = params.copy()
    params['QueueUrl'] = queue_url
    service_model = self.sqs_connection.meta.service_model
    operation_model = service_model.operation_model(operation)
    url = self.sqs_connection._endpoint.host
    headers = {}
    json_version = operation_model.metadata['jsonVersion']
    content_type = f'application/x-amz-json-{json_version}'
    headers['Content-Type'] = content_type
    target = '{}.{}'.format(operation_model.metadata['targetPrefix'], operation_model.name)
    headers['X-Amz-Target'] = target
    param_payload = {'data': json.dumps(params), 'headers': headers}
    method = operation_model.http.get('method', Serializer.DEFAULT_METHOD)
    return AWSRequest(method=method, url=url, **param_payload)