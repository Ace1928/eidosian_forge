import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class EventStreamJSONParser(BaseEventStreamParser, BaseJSONParser):

    def _initial_body_parse(self, body_contents):
        return self._parse_body_as_json(body_contents)