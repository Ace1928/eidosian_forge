import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _get_error_root(self, original_root):
    for child in original_root:
        if self._node_tag(child) == 'Errors':
            for errors_child in child:
                if self._node_tag(errors_child) == 'Error':
                    return errors_child
    return original_root