import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _populate_response_metadata(self, response):
    metadata = {}
    headers = response['headers']
    if 'x-amzn-requestid' in headers:
        metadata['RequestId'] = headers['x-amzn-requestid']
    elif 'x-amz-request-id' in headers:
        metadata['RequestId'] = headers['x-amz-request-id']
        metadata['HostId'] = headers.get('x-amz-id-2', '')
    return metadata