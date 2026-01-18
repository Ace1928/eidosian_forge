import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class RestXMLParser(BaseRestParser, BaseXMLResponseParser):
    EVENT_STREAM_PARSER_CLS = EventStreamXMLParser

    def _initial_body_parse(self, xml_string):
        if not xml_string:
            return ETree.Element('')
        return self._parse_xml_string_to_dom(xml_string)

    def _do_error_parse(self, response, shape):
        if response['body']:
            try:
                return self._parse_error_from_body(response)
            except ResponseParserError:
                LOG.debug('Exception caught when parsing error response body:', exc_info=True)
        return self._parse_error_from_http_status(response)

    def _parse_error_from_http_status(self, response):
        return {'Error': {'Code': str(response['status_code']), 'Message': http.client.responses.get(response['status_code'], '')}, 'ResponseMetadata': {'RequestId': response['headers'].get('x-amz-request-id', ''), 'HostId': response['headers'].get('x-amz-id-2', '')}}

    def _parse_error_from_body(self, response):
        xml_contents = response['body']
        root = self._parse_xml_string_to_dom(xml_contents)
        parsed = self._build_name_to_xml_node(root)
        self._replace_nodes(parsed)
        if root.tag == 'Error':
            metadata = self._populate_response_metadata(response)
            parsed.pop('RequestId', '')
            parsed.pop('HostId', '')
            return {'Error': parsed, 'ResponseMetadata': metadata}
        elif 'RequestId' in parsed:
            parsed['ResponseMetadata'] = {'RequestId': parsed.pop('RequestId')}
        default = {'Error': {'Message': '', 'Code': ''}}
        merge_dicts(default, parsed)
        return default

    @_text_content
    def _handle_string(self, shape, text):
        text = super()._handle_string(shape, text)
        return text