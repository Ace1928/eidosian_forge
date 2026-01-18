import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _parse_body_as_xml(self, response, shape, inject_metadata=True):
    xml_contents = response['body']
    root = self._parse_xml_string_to_dom(xml_contents)
    parsed = {}
    if shape is not None:
        start = root
        if 'resultWrapper' in shape.serialization:
            start = self._find_result_wrapped_shape(shape.serialization['resultWrapper'], root)
        parsed = self._parse_shape(shape, start)
    if inject_metadata:
        self._inject_response_metadata(root, parsed)
    return parsed