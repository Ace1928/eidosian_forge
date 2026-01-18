import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _inject_error_code(self, error, response):
    body = self._initial_body_parse(response['body'])
    if 'x-amzn-errortype' in response['headers']:
        code = response['headers']['x-amzn-errortype']
        code = code.split(':')[0]
        error['Error']['Code'] = code
    elif 'code' in body or 'Code' in body:
        error['Error']['Code'] = body.get('code', body.get('Code', ''))