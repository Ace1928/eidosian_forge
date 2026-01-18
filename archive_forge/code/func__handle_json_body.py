import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _handle_json_body(self, raw_body, shape):
    parsed_json = self._parse_body_as_json(raw_body)
    return self._parse_shape(shape, parsed_json)