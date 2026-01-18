import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _replace_nodes(self, parsed):
    for key, value in parsed.items():
        if list(value):
            sub_dict = self._build_name_to_xml_node(value)
            parsed[key] = self._replace_nodes(sub_dict)
        else:
            parsed[key] = value.text
    return parsed