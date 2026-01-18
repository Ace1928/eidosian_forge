import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _get_text_content(self, shape, node_or_string):
    if hasattr(node_or_string, 'text'):
        text = node_or_string.text
        if text is None:
            text = ''
    else:
        text = node_or_string
    return func(self, shape, text)