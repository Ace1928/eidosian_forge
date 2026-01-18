import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _add_modeled_parse(self, response, shape, final_parsed):
    if shape is None:
        return final_parsed
    member_shapes = shape.members
    self._parse_non_payload_attrs(response, shape, member_shapes, final_parsed)
    self._parse_payload(response, shape, member_shapes, final_parsed)