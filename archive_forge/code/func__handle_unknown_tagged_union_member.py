import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _handle_unknown_tagged_union_member(self, tag):
    return {'SDK_UNKNOWN_MEMBER': {'name': tag}}