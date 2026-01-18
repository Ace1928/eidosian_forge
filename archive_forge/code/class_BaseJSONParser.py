import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
class BaseJSONParser(ResponseParser):

    def _handle_structure(self, shape, value):
        final_parsed = {}
        if shape.is_document_type:
            final_parsed = value
        else:
            member_shapes = shape.members
            if value is None:
                return None
            final_parsed = {}
            if self._has_unknown_tagged_union_member(shape, value):
                tag = self._get_first_key(value)
                return self._handle_unknown_tagged_union_member(tag)
            for member_name in member_shapes:
                member_shape = member_shapes[member_name]
                json_name = member_shape.serialization.get('name', member_name)
                raw_value = value.get(json_name)
                if raw_value is not None:
                    final_parsed[member_name] = self._parse_shape(member_shapes[member_name], raw_value)
        return final_parsed

    def _handle_map(self, shape, value):
        parsed = {}
        key_shape = shape.key
        value_shape = shape.value
        for key, value in value.items():
            actual_key = self._parse_shape(key_shape, key)
            actual_value = self._parse_shape(value_shape, value)
            parsed[actual_key] = actual_value
        return parsed

    def _handle_blob(self, shape, value):
        return self._blob_parser(value)

    def _handle_timestamp(self, shape, value):
        return self._timestamp_parser(value)

    def _do_error_parse(self, response, shape):
        body = self._parse_body_as_json(response['body'])
        error = {'Error': {'Message': '', 'Code': ''}, 'ResponseMetadata': {}}
        headers = response['headers']
        error['Error']['Message'] = body.get('message', body.get('Message', ''))
        response_code = response.get('status_code')
        code = body.get('__type', response_code and str(response_code))
        if code is not None:
            if '#' in code:
                code = code.rsplit('#', 1)[1]
            if 'x-amzn-query-error' in headers:
                code = self._do_query_compatible_error_parse(code, headers, error)
            error['Error']['Code'] = code
        self._inject_response_metadata(error, response['headers'])
        return error

    def _do_query_compatible_error_parse(self, code, headers, error):
        """
        Error response may contain an x-amzn-query-error header to translate
        errors codes from former `query` services into `json`. We use this to
        do our lookup in the errorfactory for modeled errors.
        """
        query_error = headers['x-amzn-query-error']
        query_error_components = query_error.split(';')
        if len(query_error_components) == 2 and query_error_components[0]:
            error['Error']['QueryErrorCode'] = code
            error['Error']['Type'] = query_error_components[1]
            return query_error_components[0]
        return code

    def _inject_response_metadata(self, parsed, headers):
        if 'x-amzn-requestid' in headers:
            parsed.setdefault('ResponseMetadata', {})['RequestId'] = headers['x-amzn-requestid']

    def _parse_body_as_json(self, body_contents):
        if not body_contents:
            return {}
        body = body_contents.decode(self.DEFAULT_ENCODING)
        try:
            original_parsed = json.loads(body)
            return original_parsed
        except ValueError:
            return {'message': body}