import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class RestJSONParser(BaseRestParser, BaseJSONParser):
    EVENT_STREAM_PARSER_CLS = EventStreamJSONParser

    def _initial_body_parse(self, body_contents):
        return self._parse_body_as_json(body_contents)

    def _do_error_parse(self, response, shape):
        error = super()._do_error_parse(response, shape)
        self._inject_error_code(error, response)
        return error

    def _inject_error_code(self, error, response):
        body = self._initial_body_parse(response['body'])
        if 'x-amzn-errortype' in response['headers']:
            code = response['headers']['x-amzn-errortype']
            code = code.split(':')[0]
            error['Error']['Code'] = code
        elif 'code' in body or 'Code' in body:
            error['Error']['Code'] = body.get('code', body.get('Code', ''))

    def _handle_integer(self, shape, value):
        return int(value)
    _handle_long = _handle_integer