import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class QueryParser(BaseXMLResponseParser):

    def _do_error_parse(self, response, shape):
        xml_contents = response['body']
        root = self._parse_xml_string_to_dom(xml_contents)
        parsed = self._build_name_to_xml_node(root)
        self._replace_nodes(parsed)
        if 'Errors' in parsed:
            parsed.update(parsed.pop('Errors'))
        if 'RequestId' in parsed:
            parsed['ResponseMetadata'] = {'RequestId': parsed.pop('RequestId')}
        return parsed

    def _do_modeled_error_parse(self, response, shape):
        return self._parse_body_as_xml(response, shape, inject_metadata=False)

    def _do_parse(self, response, shape):
        return self._parse_body_as_xml(response, shape, inject_metadata=True)

    def _parse_body_as_xml(self, response, shape, inject_metadata=True):
        xml_contents = response['body']
        root = self._parse_xml_string_to_dom(xml_contents)
        parsed = {}
        if shape is not None:
            start = root
            if 'resultWrapper' in shape.serialization:
                start = self._find_result_wrapped_shape(shape.serialization['resultWrapper'], root)
            parsed = self._parse_shape(shape, start)
        if inject_metadata:
            self._inject_response_metadata(root, parsed)
        return parsed

    def _find_result_wrapped_shape(self, element_name, xml_root_node):
        mapping = self._build_name_to_xml_node(xml_root_node)
        return mapping[element_name]

    def _inject_response_metadata(self, node, inject_into):
        mapping = self._build_name_to_xml_node(node)
        child_node = mapping.get('ResponseMetadata')
        if child_node is not None:
            sub_mapping = self._build_name_to_xml_node(child_node)
            for key, value in sub_mapping.items():
                sub_mapping[key] = value.text
            inject_into['ResponseMetadata'] = sub_mapping