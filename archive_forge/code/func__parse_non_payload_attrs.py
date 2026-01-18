import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _parse_non_payload_attrs(self, response, shape, member_shapes, final_parsed):
    headers = response['headers']
    for name in member_shapes:
        member_shape = member_shapes[name]
        location = member_shape.serialization.get('location')
        if location is None:
            continue
        elif location == 'statusCode':
            final_parsed[name] = self._parse_shape(member_shape, response['status_code'])
        elif location == 'headers':
            final_parsed[name] = self._parse_header_map(member_shape, headers)
        elif location == 'header':
            header_name = member_shape.serialization.get('name', name)
            if header_name in headers:
                final_parsed[name] = self._parse_shape(member_shape, headers[header_name])