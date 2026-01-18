import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _parse_error_from_http_status(self, response):
    return {'Error': {'Code': str(response['status_code']), 'Message': http.client.responses.get(response['status_code'], '')}, 'ResponseMetadata': {'RequestId': response['headers'].get('x-amz-request-id', ''), 'HostId': response['headers'].get('x-amz-id-2', '')}}