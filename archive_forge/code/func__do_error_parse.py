import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _do_error_parse(self, response, shape):
    if response['body']:
        try:
            return self._parse_error_from_body(response)
        except ResponseParserError:
            LOG.debug('Exception caught when parsing error response body:', exc_info=True)
    return self._parse_error_from_http_status(response)