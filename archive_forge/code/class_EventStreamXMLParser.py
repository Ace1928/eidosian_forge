import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class EventStreamXMLParser(BaseEventStreamParser, BaseXMLResponseParser):

    def _initial_body_parse(self, xml_string):
        if not xml_string:
            return ETree.Element('')
        return self._parse_xml_string_to_dom(xml_string)