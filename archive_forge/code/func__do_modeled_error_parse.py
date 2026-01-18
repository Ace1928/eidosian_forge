import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _do_modeled_error_parse(self, response, shape):
    final_parsed = {}
    self._add_modeled_parse(response, shape, final_parsed)
    return final_parsed