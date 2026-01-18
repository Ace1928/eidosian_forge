import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _is_modeled_error_shape(self, shape):
    return shape is not None and shape.metadata.get('exception', False)