import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _create_event_stream(self, response, shape):
    parser = self._event_stream_parser
    name = response['context'].get('operation_name')
    return EventStream(response['body'], shape, parser, name)