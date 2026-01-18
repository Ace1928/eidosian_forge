import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _parse_xml_string_to_dom(self, xml_string):
    try:
        parser = ETree.XMLParser(target=ETree.TreeBuilder(), encoding=self.DEFAULT_ENCODING)
        parser.feed(xml_string)
        root = parser.close()
    except XMLParseError as e:
        raise ResponseParserError('Unable to parse response (%s), invalid XML received. Further retries may succeed:\n%s' % (e, xml_string))
    return root