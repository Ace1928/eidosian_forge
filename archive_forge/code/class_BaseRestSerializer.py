import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
class BaseRestSerializer(Serializer):
    """Base class for rest protocols.

    The only variance between the various rest protocols is the
    way that the body is serialized.  All other aspects (headers, uri, etc.)
    are the same and logic for serializing those aspects lives here.

    Subclasses must implement the ``_serialize_body_params`` method.

    """
    QUERY_STRING_TIMESTAMP_FORMAT = 'iso8601'
    HEADER_TIMESTAMP_FORMAT = 'rfc822'
    KNOWN_LOCATIONS = ['uri', 'querystring', 'header', 'headers']

    def serialize_to_request(self, parameters, operation_model):
        serialized = self._create_default_request()
        serialized['method'] = operation_model.http.get('method', self.DEFAULT_METHOD)
        shape = operation_model.input_shape
        if shape is None:
            serialized['url_path'] = operation_model.http['requestUri']
            return serialized
        shape_members = shape.members
        partitioned = {'uri_path_kwargs': self.MAP_TYPE(), 'query_string_kwargs': self.MAP_TYPE(), 'body_kwargs': self.MAP_TYPE(), 'headers': self.MAP_TYPE()}
        for param_name, param_value in parameters.items():
            if param_value is None:
                continue
            self._partition_parameters(partitioned, param_name, param_value, shape_members)
        serialized['url_path'] = self._render_uri_template(operation_model.http['requestUri'], partitioned['uri_path_kwargs'])
        if 'authPath' in operation_model.http:
            serialized['auth_path'] = self._render_uri_template(operation_model.http['authPath'], partitioned['uri_path_kwargs'])
        serialized['query_string'] = partitioned['query_string_kwargs']
        if partitioned['headers']:
            serialized['headers'] = partitioned['headers']
        self._serialize_payload(partitioned, parameters, serialized, shape, shape_members)
        self._serialize_content_type(serialized, shape, shape_members)
        host_prefix = self._expand_host_prefix(parameters, operation_model)
        if host_prefix is not None:
            serialized['host_prefix'] = host_prefix
        return serialized

    def _render_uri_template(self, uri_template, params):
        encoded_params = {}
        for template_param in re.findall('{(.*?)}', uri_template):
            if template_param.endswith('+'):
                encoded_params[template_param] = percent_encode(params[template_param[:-1]], safe='/~')
            else:
                encoded_params[template_param] = percent_encode(params[template_param])
        return uri_template.format(**encoded_params)

    def _serialize_payload(self, partitioned, parameters, serialized, shape, shape_members):
        payload_member = shape.serialization.get('payload')
        if self._has_streaming_payload(payload_member, shape_members):
            body_payload = parameters.get(payload_member, b'')
            body_payload = self._encode_payload(body_payload)
            serialized['body'] = body_payload
        elif payload_member is not None:
            body_params = parameters.get(payload_member)
            if body_params is not None:
                serialized['body'] = self._serialize_body_params(body_params, shape_members[payload_member])
            else:
                serialized['body'] = self._serialize_empty_body()
        elif partitioned['body_kwargs']:
            serialized['body'] = self._serialize_body_params(partitioned['body_kwargs'], shape)
        elif self._requires_empty_body(shape):
            serialized['body'] = self._serialize_empty_body()

    def _serialize_empty_body(self):
        return b''

    def _serialize_content_type(self, serialized, shape, shape_members):
        """
        Some protocols require varied Content-Type headers
        depending on user input. This allows subclasses to apply
        this conditionally.
        """
        pass

    def _requires_empty_body(self, shape):
        """
        Some protocols require a specific body to represent an empty
        payload. This allows subclasses to apply this conditionally.
        """
        return False

    def _has_streaming_payload(self, payload, shape_members):
        """Determine if payload is streaming (a blob or string)."""
        return payload is not None and shape_members[payload].type_name in ('blob', 'string')

    def _encode_payload(self, body):
        if isinstance(body, str):
            return body.encode(self.DEFAULT_ENCODING)
        return body

    def _partition_parameters(self, partitioned, param_name, param_value, shape_members):
        member = shape_members[param_name]
        location = member.serialization.get('location')
        key_name = member.serialization.get('name', param_name)
        if location == 'uri':
            partitioned['uri_path_kwargs'][key_name] = param_value
        elif location == 'querystring':
            if isinstance(param_value, dict):
                partitioned['query_string_kwargs'].update(param_value)
            elif isinstance(param_value, bool):
                bool_str = str(param_value).lower()
                partitioned['query_string_kwargs'][key_name] = bool_str
            elif member.type_name == 'timestamp':
                timestamp_format = member.serialization.get('timestampFormat', self.QUERY_STRING_TIMESTAMP_FORMAT)
                timestamp = self._convert_timestamp_to_str(param_value, timestamp_format)
                partitioned['query_string_kwargs'][key_name] = timestamp
            else:
                partitioned['query_string_kwargs'][key_name] = param_value
        elif location == 'header':
            shape = shape_members[param_name]
            if not param_value and shape.type_name == 'list':
                return
            value = self._convert_header_value(shape, param_value)
            partitioned['headers'][key_name] = str(value)
        elif location == 'headers':
            header_prefix = key_name
            self._do_serialize_header_map(header_prefix, partitioned['headers'], param_value)
        else:
            partitioned['body_kwargs'][param_name] = param_value

    def _do_serialize_header_map(self, header_prefix, headers, user_input):
        for key, val in user_input.items():
            full_key = header_prefix + key
            headers[full_key] = val

    def _serialize_body_params(self, params, shape):
        raise NotImplementedError('_serialize_body_params')

    def _convert_header_value(self, shape, value):
        if shape.type_name == 'timestamp':
            datetime_obj = parse_to_aware_datetime(value)
            timestamp = calendar.timegm(datetime_obj.utctimetuple())
            timestamp_format = shape.serialization.get('timestampFormat', self.HEADER_TIMESTAMP_FORMAT)
            return self._convert_timestamp_to_str(timestamp, timestamp_format)
        elif shape.type_name == 'list':
            converted_value = [self._convert_header_value(shape.member, v) for v in value if v is not None]
            return ','.join(converted_value)
        elif is_json_value_header(shape):
            return self._get_base64(json.dumps(value, separators=(',', ':')))
        else:
            return value